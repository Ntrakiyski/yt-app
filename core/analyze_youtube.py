import yt_dlp
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

class YouTubeAnalyzer:
    """Analyzes YouTube URLs (video, playlist, channel) to extract metadata."""

    def __init__(self):
        self.ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,  # Faster for playlists/channels, get item URLs
            'forcejson': True, # Output metadata as JSON
            # 'skip_download': True, # Implied by extract_flat and forcejson for metadata
            'dump_single_json': True, # Get metadata for single video as JSON
            'logger': logger # Use our own logger for yt-dlp messages
        }

    def _extract_info(self, url: str) -> Optional[Dict[str, Any]]:
        """Helper to extract information using yt-dlp."""
        try:
            with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                result = ydl.extract_info(url, download=False)
                return result
        except yt_dlp.utils.DownloadError as e:
            # Handle cases like private/deleted videos, invalid URLs gracefully
            logger.warning(f"Could not extract info for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred while extracting info for {url}: {e}")
            return None

    def analyze_url(self, url: str) -> Optional[Dict[str, Any]]:
        """Analyzes any YouTube URL (video, playlist, channel) and returns structured metadata."""
        raw_info = self._extract_info(url)
        if not raw_info:
            return None

        url_type = raw_info.get('_type', 'video') # Default to video if _type is missing
        
        if url_type == 'playlist' or 'entries' in raw_info: # Channel info also comes as a playlist of its videos
            return self._parse_playlist_or_channel_info(raw_info)
        else:
            return self._parse_video_info(raw_info)

    def _parse_datetime(self, date_str: Optional[str]) -> Optional[str]:
        """Parses YYYYMMDD date string to ISO 8601 format."""
        if not date_str:
            return None
        try:
            return datetime.strptime(date_str, '%Y%m%d').isoformat() + "Z"
        except ValueError:
            return date_str # Return original if parsing fails

    def _parse_video_info(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parses metadata for a single video."""
        return {
            "type": "video",
            "id": video_data.get('id'),
            "title": video_data.get('title'),
            "url": video_data.get('webpage_url', f"https://www.youtube.com/watch?v={video_data.get('id')}"),
            "description": video_data.get('description'),
            "duration": video_data.get('duration'),
            "duration_string": video_data.get('duration_string'),
            "view_count": video_data.get('view_count'),
            "like_count": video_data.get('like_count'),
            "comment_count": video_data.get('comment_count'),
            "age_limit": video_data.get('age_limit'),
            "upload_date": self._parse_datetime(video_data.get('upload_date')),
            "uploader": video_data.get('uploader'),
            "uploader_id": video_data.get('uploader_id'),
            "uploader_url": video_data.get('uploader_url'),
            "channel_id": video_data.get('channel_id'),
            "channel_url": video_data.get('channel_url'),
            "thumbnail_url": video_data.get('thumbnail'),
            "categories": video_data.get('categories'),
            "tags": video_data.get('tags'),
            "live_status": video_data.get('live_status'),
             # Chapters might be under 'chapters' or nested
            "chapters": video_data.get('chapters', []) 
        }

    def _parse_playlist_or_channel_info(self, playlist_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parses metadata for a playlist or channel."""
        # Determine if it's a channel based on common patterns or original URL type if available
        # yt-dlp might return _type: 'playlist' for channel URLs when using extract_flat
        # Heuristic: if 'uploader' info is missing at playlist level, might be a channel listing
        playlist_type = "channel" if not playlist_data.get('uploader') and playlist_data.get('_type') == 'playlist' else "playlist"
        
        # For channels, the title might be the channel name. For playlists, it's the playlist title.
        # yt-dlp gives channel name as 'uploader' when listing videos from a channel URL, or 'title' of the playlist itself.
        name = playlist_data.get('title')
        uploader_name = playlist_data.get('uploader') # Present for actual playlists
        if playlist_type == "channel" and not name: # If it's a channel and title is missing, it might be in uploader of first video
             if playlist_data.get('entries') and playlist_data['entries'][0].get('uploader'):
                 name = playlist_data['entries'][0]['uploader']
        
        entries = []
        if playlist_data.get('entries'):
            for video_entry_data in playlist_data['entries']:
                if video_entry_data: # Ensure entry is not None
                    # These are flat entries, so they contain basic info
                    entries.append({
                        "id": video_entry_data.get('id'),
                        "title": video_entry_data.get('title'),
                        "url": video_entry_data.get('url'), # This is the direct video URL
                        "duration": video_entry_data.get('duration'),
                        "uploader": video_entry_data.get('uploader') # Uploader of the video
                    })
        
        return {
            "type": playlist_type,
            "id": playlist_data.get('id'),
            "title": name, # Playlist title or Channel name
            "url": playlist_data.get('webpage_url'),
            "uploader": uploader_name, # Uploader of the playlist (if it's a playlist by a user)
            "uploader_id": playlist_data.get('uploader_id'),
            "uploader_url": playlist_data.get('uploader_url'),
            "description": playlist_data.get('description'), # Playlist description
            "video_count": playlist_data.get('playlist_count', len(entries)),
            "view_count": playlist_data.get('view_count'), # View count of the playlist itself, if available
            "videos": entries
        }

# Example Usage:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    analyzer = YouTubeAnalyzer()

    test_urls = {
        "video": "https://www.youtube.com/watch?v=jNQXAC9IVRw", # Example: ME AT THE ZOO
        "short_playlist": "https://www.youtube.com/playlist?list=PLMC9KNkIncKtPzgY-5rmFqZFvl5aiHr_J",
        "channel_videos": "https://www.youtube.com/@LinusTechTips/videos",
        "channel_custom": "https://www.youtube.com/c/LinusTechTips",
        "channel_user": "https://www.youtube.com/user/LinusTechTips",
        "invalid_url": "https://www.youtube.com/nonexistentvideo",
        "private_video": "https://www.youtube.com/watch?v=askdjnPrivateVideo", # Hypothetical
        "channel_with_at": "https://www.youtube.com/@OpenAI/videos"
    }

    for name, url in test_urls.items():
        print(f"\n--- Analyzing {name}: {url} ---")
        metadata = analyzer.analyze_url(url)
        if metadata:
            print(f"Type: {metadata.get('type')}")
            print(f"Title: {metadata.get('title')}")
            if metadata.get('videos'):
                print(f"Video count: {len(metadata['videos'])}")
                # print(f"First video: {metadata['videos'][0] if metadata['videos'] else 'N/A'}")
            # import json
            # print(json.dumps(metadata, indent=2))
        else:
            print("Could not retrieve metadata.") 