import yt_dlp
from pathlib import Path
import logging
import re
import time

# Configure logging for this module
logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG) # Uncomment for detailed yt-dlp debugging

class YTDLPLogger:
    def debug(self, msg):
        # For Sentry, targeting specific messages
        if 'sentry' in msg.lower() or 'error' in msg.lower() or 'warning' in msg.lower():
            logger.debug(msg) # Or logger.info(msg) if you want them more visible
        # else: pass # Ignore other debug messages from yt-dlp unless needed
    def warning(self, msg):
        logger.warning(msg)
    def error(self, msg):
        logger.error(msg)
    def info(self, msg):
        logger.info(msg) # yt-dlp's "info" can be quite verbose, e.g. download progress

class YouTubeDownloaderHelper:
    """Helper class to manage yt-dlp options and downloads"""
    def __init__(self, download_dir: str, audio_only: bool = False):
        self.download_dir = Path(download_dir)
        # Ensure the main download directory exists
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.audio_only = audio_only
        self.ydl_opts = self._get_ydl_opts()

    def _get_ydl_opts(self, subfolder: str = None) -> dict:
        """Generate yt-dlp options dictionary"""
        output_path_template = self.download_dir
        if subfolder:
            output_path_template = self.download_dir / subfolder
            output_path_template.mkdir(parents=True, exist_ok=True)

        # Define base options
        opts = {
            'format': 'bestaudio/best' if self.audio_only else 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': str(output_path_template / '%(title)s [%(id)s].%(ext)s'),
            'noplaylist': True, # Handled by get_video_urls for playlists/channels
            'ignoreerrors': True, # Skip problematic videos in playlists/channels
            'merge_output_format': 'mp4' if not self.audio_only else None,
            'postprocessors': [],
            'logger': YTDLPLogger(),
            'progress_hooks': [self._progress_hook],
            'quiet': True, # Suppress direct output, use logger and progress hooks
            'no_warnings': False, # Get warnings through logger
            'verbose': False, # Set to True for extreme debugging of yt-dlp
            # Other useful options:
            # 'writesubtitles': True, 'writeautomaticsub': True, 'subtitleslangs': ['en'],
            # 'writethumbnail': True,
            # 'restrictfilenames': True,
            # 'sleep_interval': 2, 'max_sleep_interval': 5, # To be polite
            # 'download_archive': str(self.download_dir / 'downloaded_archive.txt') # Track downloaded videos
        }

        if self.audio_only:
            opts['postprocessors'].append({
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            })
            opts['outtmpl'] = str(output_path_template / '%(title)s [%(id)s].%(ext)s') # yt-dlp handles ext for audio
        
        return opts

    def _progress_hook(self, d):
        if d['status'] == 'downloading':
            filename = d.get('filename', d.get('info_dict', {}).get('_filename', 'Unknown file'))
            #logger.info(f"Downloading {Path(filename).name}: {d['_percent_str']} of {d['_total_bytes_str_or_percent']} at {d['_speed_str']}")
        elif d['status'] == 'finished':
            filename = d.get('filename', d.get('info_dict', {}).get('_filename', 'Unknown file'))
            #logger.info(f"Finished downloading {Path(filename).name}")
        elif d['status'] == 'error':
            filename = d.get('filename', d.get('info_dict', {}).get('_filename', 'Unknown file'))
            logger.error(f"Error downloading {Path(filename).name}")

    def extract_info(self, url: str, download: bool = True, subfolder: str = None):
        """Download or extract info for a given URL"""
        current_opts = self._get_ydl_opts(subfolder=subfolder)
        try:
            with yt_dlp.YoutubeDL(current_opts) as ydl:
                info = ydl.extract_info(url, download=download)
                # If download is True, ydl.extract_info returns info_dict
                # The actual downloaded file path is in info_dict.get('requested_downloads')[0].get('filepath')
                # or constructed from outtmpl if single file.
                if download:
                    # yt-dlp >= 2023.06.22 uses `requested_downloads` for file path info
                    if 'requested_downloads' in info and info['requested_downloads']:
                        # For merged formats, the final file is often the first one.
                        # This can be complex if multiple files are part of the download (e.g. video + audio before merge)
                        # We are interested in the final output file.
                        # The `outtmpl` determines the final name.
                        # Let's try to reconstruct the expected path based on `outtmpl` and info.
                        
                        # Simplification: assume single primary output or rely on known naming pattern
                        # If audio_only, extension becomes mp3 due to postprocessor
                        file_ext = 'mp3' if self.audio_only else info.get('ext', 'mp4') 
                        
                        # Build the filename based on the outtmpl pattern. This is a bit fragile.
                        # A more robust way is to scan the directory for the newest file matching the ID.
                        # For now, let's try constructing from title and ID.
                        title = info.get('title', 'default_title')
                        video_id = info.get('id', 'default_id')
                        
                        # Sanitize title for filename
                        sane_title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', title)
                        sane_title = sane_title[:150] # Limit length
                        
                        expected_filename = f"{sane_title} [{video_id}].{file_ext}"
                        expected_path = (output_path_template if subfolder else self.download_dir) / expected_filename
                        
                        # Check if this path exists, otherwise fall back to what yt-dlp reports
                        if expected_path.exists():
                            info['_final_filepath'] = str(expected_path)
                            return info
                        else: # Fallback if constructed path isn't found
                            # This part is tricky because `info` after download might not directly give the *final* merged path
                            # readily, especially if `outtmpl` is complex or uses playlist indexing etc.
                            # The `filename` key in `d` of progress_hooks usually points to the final file.
                            # If not available here, we might need to scan or use a more direct way from ydl object if possible.
                            # For now, if the constructed path isn't found, we'll signal it
                            logger.warning(f"Constructed path {expected_path} not found. Falling back.")
                            # Try to get from requested_downloads if available
                            if info.get('requested_downloads') and info['requested_downloads'][0].get('filepath'):
                                info['_final_filepath'] = info['requested_downloads'][0]['filepath']
                                return info
                            # As a last resort, if the file extension matches, use the info's filepath
                            if Path(info.get('filepath', '')).suffix == f'.{file_ext}':
                                info['_final_filepath'] = info.get('filepath')
                                return info
                            
                            # If still no path, this download likely failed or path is unknown.
                            logger.error(f"Could not determine final file path for {info.get('title')}")
                            return None # Indicate failure to determine path
                return info # if download=False
        except yt_dlp.utils.DownloadError as e:
            logger.error(f"yt-dlp DownloadError for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Generic error in extract_info for {url}: {e}")
            return None


class YouTubeChannelDownloader:
    """Downloads videos from YouTube channels, playlists, or single URLs"""
    def __init__(self, download_dir: str):
        self.download_dir = Path(download_dir)
        # Ensure base download directory exists
        self.download_dir.mkdir(parents=True, exist_ok=True)
        # Sub-folders for transcriptions will be handled by VideoTranscriber or main app logic
        self.yt_dlp_helper = None # Will be initialized per download type

    def _get_channel_name_from_url(self, url: str) -> str:
        """Attempt to extract a readable channel name from URL for subfolder creation"""
        try:
            match = re.search(r'(?:/c/|/user/|/@)([^/]+)', url, re.IGNORECASE)
            if match:
                return re.sub(r'[^a-zA-Z0-9_-]', '_', match.group(1)) # Sanitize
            # Fallback for /channel/ URLs that might not have a custom name
            match_channel_id = re.search(r'/channel/([a-zA-Z0-9_-]+)', url, re.IGNORECASE)
            if match_channel_id:
                return match_channel_id.group(1) # Use channel ID if no custom name
        except Exception:
            pass
        return "default_channel"

    def _get_playlist_name_from_info(self, info: dict) -> str:
        """Extract playlist name from info dict, or create default"""
        playlist_title = info.get('playlist_title', info.get('title', None))
        if playlist_title:
            return re.sub(r'[^a-zA-Z0-9_-]', '_', playlist_title)[:100] # Sanitize and shorten
        return f"playlist_{info.get('playlist_id', 'default_playlist')}"

    def get_video_urls(self, url: str, max_videos: int = 5) -> List[str]:
        """Get list of video URLs from a channel or playlist URL"""
        # Basic opts for fetching playlist/channel info without downloading media
        info_opts = {
            'extract_flat': 'in_playlist', # Get direct video URLs
            'playlistend': max_videos if max_videos > 0 else None,
            'ignoreerrors': True,
            'quiet': True,
            'logger': YTDLPLogger(),
        }
        video_urls = []
        try:
            with yt_dlp.YoutubeDL(info_opts) as ydl:
                result = ydl.extract_info(url, download=False)
                if result and 'entries' in result:
                    for entry in result['entries']:
                        if entry and entry.get('url'):
                            video_urls.append(entry['url'])
                        if max_videos > 0 and len(video_urls) >= max_videos:
                            break # Respect max_videos limit early
            return video_urls
        except Exception as e:
            logger.error(f"Error fetching video URLs from {url}: {e}")
            return []

    def download_video(self, url: str, audio_only: bool = False, subfolder_name: str = None) -> Optional[str]:
        """Downloads a single video or audio.
        Returns the path to the downloaded file if successful, else None.
        """
        self.yt_dlp_helper = YouTubeDownloaderHelper(str(self.download_dir), audio_only=audio_only)
        
        # Determine subfolder for organization if not explicitly provided.
        # This helps keep downloads from different sources (channels/playlists) separate.
        # For single videos, they might go into the root of download_dir or a generic subfolder.
        final_subfolder = subfolder_name
        
        # If it's a playlist or channel URL, yt-dlp usually handles subdirectories with `%(playlist_title)s`
        # in `outtmpl`. For single videos, we might not have this context directly here.
        # The `YouTubeDownloaderHelper` will use the subfolder if passed.

        logger.info(f"Starting download for: {url} (Audio only: {audio_only})")
        info = self.yt_dlp_helper.extract_info(url, download=True, subfolder=final_subfolder)
        
        if info and info.get('_final_filepath') and Path(info['_final_filepath']).exists():
            final_path = Path(info['_final_filepath'])
            logger.info(f"Successfully downloaded and confirmed: {final_path}")
            return str(final_path)
        elif info and info.get('filepath') and Path(info['filepath']).exists():
            # Fallback if _final_filepath was not set but filepath exists (e.g. non-merged audio)
            final_path = Path(info['filepath'])
            logger.warning(f"Downloaded via fallback filepath: {final_path}")
            return str(final_path)
        else:
            logger.error(f"Download failed or file path not found for URL: {url}")
            # Attempt to find the file if title and ID are known (last resort)
            if info and info.get('title') and info.get('id'):
                sane_title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', info['title'])[:150]
                video_id = info['id']
                expected_ext = 'mp3' if audio_only else 'mp4'
                
                # Search in the base download_dir or specified subfolder
                search_dir = self.download_dir / final_subfolder if final_subfolder else self.download_dir
                
                # Try common naming patterns yt-dlp might produce
                possible_patterns = [
                    f"{sane_title} [{video_id}].{expected_ext}",
                    f"{sane_title}.{expected_ext}", # if ID not in filename by some option
                    f"{video_id}.{expected_ext}" # if only ID used
                ]
                for pattern in possible_patterns:
                    potential_path = search_dir / pattern
                    if potential_path.exists():
                        logger.warning(f"Found file via manual search: {potential_path}")
                        return str(potential_path)
                
                # If not found, try a broader search for any file with the ID in its name
                # This is slow and should be a last resort.
                # This is also too broad, disabled for now.
                # for item in search_dir.rglob(f"*{video_id}*.{expected_ext}"):
                #     if item.is_file():
                #         logger.warning(f"Found file via broad search: {item}")
                #         return str(item)

            logger.error(f"Could not confirm downloaded file for {url}. Check logs and download directory.")
            return None

    def download_channel_videos(self, channel_url: str, max_videos: int, audio_only: bool = False):
        """Downloads videos from a YouTube channel"""
        logger.info(f"Starting download for channel: {channel_url}, max_videos={max_videos}")
        channel_name_for_folder = self._get_channel_name_from_url(channel_url)
        
        video_urls = self.get_video_urls(channel_url, max_videos)
        downloaded_files = []
        if not video_urls:
            logger.warning(f"No video URLs found for channel {channel_url}")
            return []
            
        for url in video_urls:
            file_path = self.download_video(url, audio_only=audio_only, subfolder_name=channel_name_for_folder)
            if file_path:
                downloaded_files.append(file_path)
            time.sleep(1) # Be polite
        return downloaded_files

    def download_playlist_videos(self, playlist_url: str, max_videos: int, audio_only: bool = False):
        """Downloads videos from a YouTube playlist"""
        logger.info(f"Starting download for playlist: {playlist_url}, max_videos={max_videos}")        
        
        # Get playlist title for folder naming using a preliminary info extraction
        # This is a bit inefficient but helps organize files well.
        playlist_name_for_folder = "default_playlist"
        try:
            temp_info_opts = {'extract_flat': True, 'quiet': True, 'logger': YTDLPLogger()}
            with yt_dlp.YoutubeDL(temp_info_opts) as ydl:
                playlist_info = ydl.extract_info(playlist_url, download=False)
                if playlist_info:
                    playlist_name_for_folder = self._get_playlist_name_from_info(playlist_info)
        except Exception as e:
            logger.warning(f"Could not extract playlist title for folder name: {e}. Using default.")

        video_urls = self.get_video_urls(playlist_url, max_videos)
        downloaded_files = []
        if not video_urls:
            logger.warning(f"No video URLs found for playlist {playlist_url}")
            return []

        for url in video_urls:
            file_path = self.download_video(url, audio_only=audio_only, subfolder_name=playlist_name_for_folder)
            if file_path:
                downloaded_files.append(file_path)
            time.sleep(1) # Be polite
        return downloaded_files


# Example Usage (for testing module directly)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # --- Configuration ---
    # Target download directory (will be created if it doesn't exist)
    DOWNLOAD_DIRECTORY = Path.home() / "Downloads" / "TestYouTubeDownloads"
    DOWNLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    
    downloader = YouTubeChannelDownloader(download_dir=str(DOWNLOAD_DIRECTORY))

    # --- Test Cases ---
    # 1. Single Video (MP4)
    # logger.info("\n--- Testing Single Video (MP4) ---")
    # video_url = 'https://www.youtube.com/watch?v=dQw4w9WgXcQ' # Rick Astley
    # downloaded_video_path = downloader.download_video(video_url, audio_only=False)
    # if downloaded_video_path:
    #     logger.info(f"Single video downloaded to: {downloaded_video_path}")
    # else:
    #     logger.error("Single video download failed.")

    # # 2. Single Video (MP3)
    # logger.info("\n--- Testing Single Video (MP3) ---")
    # audio_url = 'https://www.youtube.com/watch?v=y2bY-sP1x3U' # Short sound effect
    # downloaded_audio_path = downloader.download_video(audio_url, audio_only=True)
    # if downloaded_audio_path:
    #     logger.info(f"Single audio downloaded to: {downloaded_audio_path}")
    # else:
    #     logger.error("Single audio download failed.")

    # # 3. Playlist (first 2 videos, MP4)
    # logger.info("\n--- Testing Playlist (MP4, Max 2) ---")
    # playlist_url = 'https://www.youtube.com/playlist?list=PLMC9KNkIncKtPzgY-5rmFqZFvl5aiHr_J' # Short example playlist
    # playlist_files_mp4 = downloader.download_playlist_videos(playlist_url, max_videos=2, audio_only=False)
    # if playlist_files_mp4:
    #     logger.info(f"Playlist videos (MP4) downloaded: {len(playlist_files_mp4)} files")
    #     for f in playlist_files_mp4: logger.info(f" - {f}")
    # else:
    #     logger.error("Playlist (MP4) download failed or no videos found.")

    # # 4. Playlist (first 2 videos, MP3)
    # logger.info("\n--- Testing Playlist (MP3, Max 2) ---")
    # playlist_files_mp3 = downloader.download_playlist_videos(playlist_url, max_videos=2, audio_only=True)
    # if playlist_files_mp3:
    #     logger.info(f"Playlist videos (MP3) downloaded: {len(playlist_files_mp3)} files")
    #     for f in playlist_files_mp3: logger.info(f" - {f}")
    # else:
    #     logger.error("Playlist (MP3) download failed or no videos found.")

    # # 5. Channel (first 1 video, MP4) - Use a channel with frequent short uploads for quick test
    # logger.info("\n--- Testing Channel (MP4, Max 1) ---")
    # channel_url = 'https://www.youtube.com/@MrBeast/videos' # Example: MrBeast
    # # channel_url = 'https://www.youtube.com/user/LinusTechTips/videos' # Alternative
    # channel_files_mp4 = downloader.download_channel_videos(channel_url, max_videos=1, audio_only=False)
    # if channel_files_mp4:
    #     logger.info(f"Channel video (MP4) downloaded: {channel_files_mp4[0]}")
    # else:
    #     logger.error("Channel (MP4) download failed or no video found.")

    logger.info("\n--- All tests complete (commented out by default) ---")
    logger.info(f"Check download directory: {DOWNLOAD_DIRECTORY}") 