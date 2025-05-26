#!/usr/bin/env python3
"""
YouTube Downloaders & Transcriber - Streamlit Web Interface (Version 2)

A comprehensive web interface for downloading YouTube videos and generating transcriptions
using OpenAI's Whisper AI model. Supports channels, playlists, and individual videos.
"""

# Comprehensive warning and error suppression
import warnings
import logging
import os
import sys
import io
from contextlib import redirect_stderr, redirect_stdout

# Suppress ALL warnings before any other imports
warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress torch-specific warnings and errors
# Remove problematic TORCH_LOGS setting
if 'TORCH_LOGS' in os.environ:
    del os.environ['TORCH_LOGS']
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Redirect stderr temporarily to suppress torch import warnings
original_stderr = sys.stderr
sys.stderr = io.StringIO()

try:
    # Import torch first to trigger any warnings in our suppressed environment
    import torch
    torch.set_num_threads(1)  # Reduce torch overhead
except ImportError:
    pass  # torch not available, that's fine

# Restore stderr
sys.stderr = original_stderr

# Configure logging to be less verbose
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.CRITICAL)
logging.getLogger("whisper").setLevel(logging.ERROR)
logging.getLogger("moviepy").setLevel(logging.ERROR)
logging.getLogger("yt_dlp").setLevel(logging.WARNING)

# Suppress specific warning categories
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*torch.*")
warnings.filterwarnings("ignore", message=".*classes.*")

import streamlit as st
import asyncio
import threading
import time
import json
import zipfile
import tempfile
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import plotly.express as px
from urllib.parse import urlparse, parse_qs

# Import core modules with error handling
try:
    from core.youtube_downloader import YouTubeChannelDownloader
    from core.video_transcriber import VideoTranscriber
except ImportError as e:
    st.error(f"❌ Failed to import core modules: {e}")
    st.error("Please ensure you're running from the correct directory and all dependencies are installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="YouTube Downloader & Transcriber v2",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
    }
    /* Limit main container width */
    .main .block-container {
        max-width: 1100px;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    /* Hide Streamlit style elements for cleaner look */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stApp > header {visibility: hidden;}
    /* Hide warning messages */
    .stAlert[data-baseweb="notification"] {display: none;}
</style>
""", unsafe_allow_html=True)

class YouTubeURLValidator:
    """Utility class for validating and parsing YouTube URLs"""
    
    @staticmethod
    def is_valid_youtube_url(url: str) -> bool:
        """Check if URL is a valid YouTube URL"""
        youtube_domains = ['youtube.com', 'youtu.be', 'www.youtube.com', 'm.youtube.com']
        try:
            parsed = urlparse(url)
            return any(domain in parsed.netloc for domain in youtube_domains)
        except:
            return False
    
    @staticmethod
    def detect_url_type(url: str) -> str:
        """Detect the type of YouTube URL (video, playlist, channel)"""
        if not YouTubeURLValidator.is_valid_youtube_url(url):
            return "invalid"
        
        parsed = urlparse(url)
        query_params = parse_qs(parsed.query)
        
        # Check for playlist - this includes URLs like /watch?v=...&list=...
        if 'list' in query_params:
            list_id = query_params['list'][0]
            # Check if it's a valid playlist (not just a video with list parameter)
            if list_id and not list_id.startswith('UU'):  # UU lists are channel uploads
                return "playlist"
        
        # Check for video
        if 'watch' in parsed.path or 'youtu.be' in parsed.netloc:
            return "video"
        
        # Check for channel
        if any(path in parsed.path for path in ['/channel/', '/user/', '/c/', '/@']):
            return "channel"
        
        return "unknown"
    
    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})',
            r'youtube\.com/embed/([a-zA-Z0-9_-]{11})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

class StreamlitWorkflowManager:
    """Manages the workflow of downloading and transcribing videos"""
    
    def __init__(self):
        self.downloader = None
        self.transcriber = None
        self.current_process = None
        self.results = {
            'downloads': [],
            'transcriptions': [],
            'errors': [],
            'total_files': 0
        }
        # Ensure base output directory exists when manager is initialized
        # This might be better placed where the config value is confirmed
        # For now, we assume config will provide a valid base string path.
    
    def initialize_components(self, config: Dict[str, Any]):
        """Initialize downloader and transcriber with configuration"""
        # Ensure base output directory defined in config exists
        base_output_dir = Path(config['output_dir'])
        base_output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize downloader
        self.downloader = YouTubeChannelDownloader(
            download_dir=str(base_output_dir) # Pass as string
        )
        
        # Initialize transcriber if needed
        if config['enable_transcription']:
            try:
                transcriptions_dir = base_output_dir / "transcriptions"
                transcriptions_dir.mkdir(parents=True, exist_ok=True)
                self.transcriber = VideoTranscriber(
                    model_size=config['whisper_model'],
                    device="auto",
                    output_dir=str(transcriptions_dir) # Pass to transcriber
                )
            except Exception as e:
                self.results['errors'].append(f"Failed to initialize transcriber: {str(e)}")
                # Don't fail completely, just disable transcription
                self.transcriber = None
    
    def download_videos(self, url: str, config: Dict[str, Any], progress_callback=None) -> List[str]:
        """Download videos and return list of file paths with organized folder structure"""
        downloaded_files = []
        base_output_dir = Path(config['output_dir'])

        try:
            if progress_callback:
                progress_callback("🔍 Analyzing URL and fetching video information...")
            
            url_type = YouTubeURLValidator.detect_url_type(url)
            
            if url_type == "video":
                # Single video download
                video_id = YouTubeURLValidator.extract_video_id(url)
                if video_id:
                    if progress_callback:
                        progress_callback(f"📥 Downloading video: {video_id}")
                    
                    success_path = self.downloader.download_video(url, audio_only=config.get('download_audio_only', False))
                    if success_path:
                        downloaded_files.append(success_path)
                    else:
                        self.results['errors'].append(f"Failed to download video: {url}")

            elif url_type in ["playlist", "channel"]:
                # Playlist or Channel download
                if progress_callback:
                    progress_callback(f"📡 Fetching video list for {url_type}: {url}")
                
                video_urls = self.downloader.get_video_urls(url, max_videos=config.get('num_videos', 5))
                self.results['total_files'] = len(video_urls)

                if not video_urls:
                    self.results['errors'].append(f"No videos found for {url_type}: {url}")
                    if progress_callback: progress_callback(f"🚫 No videos found for {url_type}: {url}")
                    return []

                for i, video_url_item in enumerate(video_urls):
                    if progress_callback:
                        progress_callback(f"📥 Downloading {url_type} video {i+1}/{len(video_urls)}: {video_url_item}")
                    
                    success_path = self.downloader.download_video(video_url_item, audio_only=config.get('download_audio_only', False))
                    if success_path:
                        downloaded_files.append(success_path)
                    else:
                        self.results['errors'].append(f"Failed to download video: {video_url_item} from {url_type} {url}")
            else:
                self.results['errors'].append(f"Unsupported URL type: {url_type} for {url}")
                if progress_callback: progress_callback(f"⚠️ Unsupported URL type: {url_type}")
        
        except Exception as e:
            self.results['errors'].append(f"Error during download: {str(e)}")
            if progress_callback: progress_callback(f"❌ Error during download: {str(e)}")
        
        return downloaded_files

    def _organize_video_file(self, video_path: str, base_output_dir: str) -> str:
        """Helper to organize downloaded files (potentially deprecated if downloader handles it)"""
        # This logic might be better inside YouTubeDownloader or simplified
        # if yt-dlp output templates are used effectively.
        try:
            source_path = Path(video_path)
            target_dir = Path(base_output_dir)
            # Assuming yt-dlp might already place it in a channel/playlist named folder
            # If not, this organization logic would be more complex.
            # For now, just ensure it's in the base_output_dir.
            if source_path.parent != target_dir:
                target_path = target_dir / source_path.name
                source_path.rename(target_path)
                return str(target_path)
            return str(source_path)
        except Exception as e:
            self.results['errors'].append(f"Error organizing file {video_path}: {str(e)}")
            return str(video_path) # return original if error

    def extract_video_metadata(self, youtube_url: str) -> Dict[str, Any]:
        """Extract metadata for a given YouTube URL using yt-dlp directly"""
        if not self.downloader: # Should be initialized by now
            return {"error": "Downloader not initialized"}
        
        if progress_callback:
            progress_callback(f"ℹ️ Extracting metadata for {youtube_url}...") # This line had a NameError for progress_callback

        try:
            # Use a method from downloader if it exists, or implement lightweight here
            # Reusing downloader's ydl_opts setup is good
            info_dict = self.downloader.yt_dlp_helper.extract_info(youtube_url, download=False)
            
            metadata = {
                "id": info_dict.get('id'),
                "title": info_dict.get('title'),
                "uploader": info_dict.get('uploader'),
                "uploader_id": info_dict.get('uploader_id'),
                "uploader_url": info_dict.get("uploader_url"),
                "channel_id": info_dict.get("channel_id"),
                "channel_url": info_dict.get("channel_url") if info_dict.get("channel_id") else info_dict.get('webpage_url').split('/channel/')[0] + '/channel/' + info_dict.get("channel_id") if info_dict.get("channel_id") else None, # simplified
                "duration": info_dict.get('duration'),
                "duration_string": info_dict.get('duration_string'),
                "description": info_dict.get('description'),
                "tags": info_dict.get('tags'),
                "view_count": info_dict.get('view_count'),
                "like_count": info_dict.get('like_count'),
                "upload_date": info_dict.get('upload_date'), # YYYYMMDD
                "webpage_url": info_dict.get('webpage_url'),
                "thumbnail_url": info_dict.get('thumbnail'),
                "categories": info_dict.get('categories'),
                "chapters": info_dict.get('chapters'),
                "playlist_title": info_dict.get('playlist_title'),
                "playlist_id": info_dict.get('playlist_id'),
                "playlist_uploader": info_dict.get('playlist_uploader')
            }
            return metadata
        except Exception as e:
            self.results['errors'].append(f"Failed to extract metadata for {youtube_url}: {str(e)}")
            return {"error": f"Failed to extract metadata: {str(e)}"}

    def transcribe_videos(self, video_files: List[str], config: Dict[str, Any], progress_callback=None, youtube_url: str = None) -> List[str]:
        """Transcribe list of video files and return paths to transcript files"""
        transcribed_files_info = [] # Store dicts with info
        
        if not self.transcriber:
            if progress_callback: progress_callback("⚠️ Transcription disabled or not initialized.")
            return []
        
        base_output_dir = Path(config['output_dir'])
        transcriptions_base_dir = base_output_dir / "transcriptions"
        # This was already created in initialize_components, but good to be sure.
        transcriptions_base_dir.mkdir(parents=True, exist_ok=True)

        for i, video_path_str in enumerate(video_files):
            video_path = Path(video_path_str)
            if progress_callback:
                progress_callback(f"🎙️ Transcribing video {i+1}/{len(video_files)}: {video_path.name}")
            
            try:
                # Determine output path for transcript (inside youtube_downloads/transcriptions/<original_video_name_without_ext>/
                # This structure helps keep things organized if multiple formats are output.
                video_file_stem = video_path.stem
                specific_transcription_dir = transcriptions_base_dir / video_file_stem
                specific_transcription_dir.mkdir(parents=True, exist_ok=True)

                # Call VideoTranscriber.transcribe
                # It should handle saving to its own configured output_dir + subfolder logic
                # For now, we pass the specific dir for this video's transcripts
                transcript_data = self.transcriber.transcribe(
                    str(video_path), 
                    output_formats=config.get('output_formats', ['txt']),
                    include_metadata=config.get('include_metadata', True),
                    timestamp_duration=config.get('timestamp_duration', 8),
                    video_metadata=self.extract_video_metadata(youtube_url) if youtube_url and config.get('include_metadata', True) else None,
                    # Pass the specific directory for this video's transcript files
                    # The VideoTranscriber's own output_dir will be its root.
                    # This is a change: VideoTranscriber should save into a subfolder named after the video stem.
                    # output_subpath = video_file_stem 
                )

                if transcript_data:
                    # VideoTranscriber.transcribe should return a list of paths to the created files
                    # or structured data that includes these paths.
                    # For now, assuming it returns a dict with file paths if successful
                    # This part needs alignment with what VideoTranscriber.transcribe returns.

                    saved_files = []
                    if "txt" in config.get('output_formats', []) and transcript_data.get("txt_file"): 
                        saved_files.append(transcript_data["txt_file"])
                    if "json" in config.get('output_formats', []) and transcript_data.get("json_file"): 
                        saved_files.append(transcript_data["json_file"])
                    # Add other formats like srt, vtt if VideoTranscriber supports them and returns paths

                    if saved_files:
                        transcribed_files_info.append({
                            "original_video_file": str(video_path),
                            "transcript_files": saved_files,
                            "metadata": transcript_data.get("metadata", {})
                        })
                        self.results['transcriptions'].extend(saved_files) # Keep flat list for overall results
                        if progress_callback: progress_callback(f"✅ Transcription complete for: {video_path.name}")
                    else:
                        self.results['errors'].append(f"Transcription completed for {video_path.name}, but no output files reported.")
                        if progress_callback: progress_callback(f"⚠️ Transcription done (no files) for: {video_path.name}")

                else:
                    self.results['errors'].append(f"Transcription failed for {video_path.name}")
                    if progress_callback: progress_callback(f"❌ Transcription failed for: {video_path.name}")

            except Exception as e:
                error_message = f"Error during transcription of {video_path.name}: {str(e)}"
                self.results['errors'].append(error_message)
                if progress_callback: progress_callback(f"❌ {error_message}")
            
            # Cleanup: Remove original video/audio if not configured to keep
            # This should be after transcription for that specific file
            is_audio_only_download = config.get('download_audio_only', False)
            
            if not config.get('keep_videos', True) and not is_audio_only_download and video_path.exists():
                try:
                    video_path.unlink()
                    if progress_callback: progress_callback(f"🗑️ Deleted video: {video_path.name}")
                except Exception as e:
                    self.results['errors'].append(f"Failed to delete video {video_path.name}: {e}")
            
            # If it was an audio-only download, keep_videos doesn't apply, 
            # use download_audio (which defaults to True if audio_only is True)
            if is_audio_only_download and not config.get('download_audio', True) and video_path.exists():
                try:
                    video_path.unlink()
                    if progress_callback: progress_callback(f"🗑️ Deleted audio: {video_path.name}")
                except Exception as e:
                    self.results['errors'].append(f"Failed to delete audio {video_path.name}: {e}")

        # If VideoTranscriber saves files to its own output_dir, we need to collect those paths
        # This needs to be clarified based on VideoTranscriber's implementation.
        # For now, assume paths are returned and collected in transcribed_files_info.
        
        return [item for sublist in [info.get("transcript_files", []) for info in transcribed_files_info] for item in sublist]


    def process_url(self, url: str, config: Dict[str, Any], progress_callback=None) -> Dict[str, List]:
        """Process a given URL: download and transcribe"""
        # Reset results for this run
        self.results = {'downloads': [], 'transcriptions': [], 'errors': [], 'total_files': 0}
        
        if progress_callback: progress_callback("🚀 Starting process...")
        
        # Initialize components based on current config (e.g., Whisper model)
        self.initialize_components(config)
        
        # Step 1: Download videos
        if progress_callback: progress_callback(" PHASE 1: DOWNLOADING VIDEOS ")
        downloaded_video_files = self.download_videos(url, config, progress_callback)
        self.results['downloads'] = downloaded_video_files
        
        if not downloaded_video_files and not self.results['errors']:
            if progress_callback: progress_callback("🤷 No videos were downloaded. Nothing to transcribe.")
        elif not downloaded_video_files and self.results['errors']:
            if progress_callback: progress_callback("⚠️ Download phase completed with errors. Check logs.")
        else:
            if progress_callback: progress_callback(f"✅ Download phase complete. {len(downloaded_video_files)} file(s) ready for transcription.")

        # Step 2: Transcribe videos (if enabled and files downloaded)
        if config['enable_transcription'] and downloaded_video_files:
            if progress_callback: progress_callback(" PHASE 2: TRANSCRIBING VIDEOS ")
            transcribed_file_paths = self.transcribe_videos(downloaded_video_files, config, progress_callback, url)
            # self.results['transcriptions'] is already updated by transcribe_videos
            if progress_callback: progress_callback(f"✅ Transcription phase complete. {len(self.results['transcriptions'])} transcript(s) generated.")
        elif config['enable_transcription'] and not downloaded_video_files:
            if progress_callback: progress_callback("ℹ️ Transcription enabled, but no videos downloaded to transcribe.")
        else:
            if progress_callback: progress_callback("ℹ️ Transcription not enabled.")

        if progress_callback: progress_callback("🏁 Process finished.")
        
        # Return all collected results
        return self.results

# --- Streamlit UI Components ---
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'workflow_manager' not in st.session_state:
        st.session_state.workflow_manager = StreamlitWorkflowManager()
    
    if 'process_status' not in st.session_state:
        st.session_state.process_status = "idle"
    
    if 'progress_messages' not in st.session_state:
        st.session_state.progress_messages = []
    
    if 'results' not in st.session_state:
        st.session_state.results = {}

def create_sidebar_config() -> Dict[str, Any]:
    """Create sidebar configuration panel"""
    st.sidebar.header("⚙️ Configuration")
    
    # Download Settings
    st.sidebar.subheader("📥 Download Settings")
    num_videos = st.sidebar.slider(
        "Number of videos", 
        min_value=1, 
        max_value=50, 
        value=2,
        help="Maximum number of videos to download"
    )
    
    output_dir = st.sidebar.text_input(
        "Output directory", 
        value="youtube_downloads",
        help="Directory to save downloaded files"
    )
    
    download_format = st.sidebar.radio(
        "Download format",
        options=["Video (MP4)", "Audio only (MP3)"],
        index=0,
        help="Choose whether to download video files or audio-only files"
    )
    
    download_videos = download_format == "Video (MP4)"
    download_audio_only = download_format == "Audio only (MP3)"
    
    if download_videos:
        keep_videos = st.sidebar.checkbox(
            "Keep video files", 
            value=True,
            help="Keep downloaded video files after transcription (unchecked = auto-delete after transcription)"
        )
    else:
        keep_videos = False
    
    download_audio = st.sidebar.checkbox(
        "Keep extracted audio files", 
        value=download_audio_only,
        help="Keep extracted audio files"
    )
    
    # Transcription Settings
    st.sidebar.subheader("🎙️ Transcription Settings")
    enable_transcription = st.sidebar.checkbox(
        "Enable transcription", 
        value=False,
        help="Generate transcripts using Whisper AI"
    )
    
    whisper_model = st.sidebar.selectbox(
        "Whisper model", 
        options=['tiny', 'base', 'small', 'medium', 'large'],
        index=0,  # Default to 'tiny' for cloud deployment
        help="Larger models are more accurate but slower. 'tiny' recommended for initial cloud tests."
    )
    
    timestamp_duration = st.sidebar.number_input(
        "Segment duration (seconds)", 
        min_value=1, 
        max_value=30, 
        value=8,
        help="Duration of each transcript segment"
    )
    
    # Output Format Settings
    st.sidebar.subheader("📄 Output Formats")
    output_formats = st.sidebar.multiselect(
        "Export formats",
        options=['txt', 'json', 'api'], # Add 'srt', 'vtt' if supported by VideoTranscriber
        default=['txt'],
        help="Choose output file formats for transcriptions. API sends data to webhook endpoint."
    )
    
    if enable_transcription and not output_formats:
        st.sidebar.warning("⚠️ Please select at least one output format for transcription")
    
    include_metadata = st.sidebar.checkbox(
        "Include metadata", 
        value=True,
        help="Include video metadata in transcripts"
    )
    
    return {
        'num_videos': num_videos,
        'output_dir': output_dir,
        'download_videos': download_videos,
        'download_audio_only': download_audio_only,
        'keep_videos': keep_videos,
        'download_audio': download_audio,
        'enable_transcription': enable_transcription,
        'whisper_model': whisper_model,
        'timestamp_duration': timestamp_duration,
        'output_formats': output_formats,
        'include_metadata': include_metadata
    }

def create_url_input_section():
    """Create URL input and validation section"""
    st.header("🔗 YouTube URL Input")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Check if an example URL was selected
        default_value = st.session_state.get('example_url', '')
        if default_value:
            # Clear the example URL after using it
            del st.session_state.example_url
        
        url = st.text_input(
            "Enter YouTube URL",
            value=default_value,
            placeholder="https://www.youtube.com/watch?v=... or channel/playlist URL",
            help="Supports individual videos, playlists, and channels"
        )
    
    with col2:
        st.write("") # Spacing
        st.write("") # Spacing
        validate_button = st.button("🔍 Validate URL", type="secondary")
    
    # URL validation and type detection
    if url or validate_button:
        if url:
            is_valid = YouTubeURLValidator.is_valid_youtube_url(url)
            url_type = YouTubeURLValidator.detect_url_type(url)
            
            if is_valid and url_type != "invalid":
                icon_map = {
                    "video": "🎥",
                    "playlist": "📋",
                    "channel": "📺",
                    "unknown": "❓"
                }
                
                st.markdown(f"""
                <div class="status-success">
                    {icon_map.get(url_type, "❓")} <strong>Valid {url_type.title()} URL</strong>
                </div>
                """, unsafe_allow_html=True)
                
                if url_type == "video":
                    video_id = YouTubeURLValidator.extract_video_id(url)
                    if video_id:
                        st.info(f"Video ID: {video_id}")
                
                return url, url_type
            else:
                st.markdown("""
                <div class="status-error">
                    ❌ <strong>Invalid YouTube URL</strong><br>
                    Please enter a valid YouTube video, playlist, or channel URL.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-warning">
                ⚠️ <strong>Please enter a YouTube URL</strong>
            </div>
            """, unsafe_allow_html=True)
    
    return None, None

def create_process_execution_section(url: str, url_type: str, config: Dict[str, Any]):
    """Create process execution section with progress tracking"""
    st.header("🚀 Process Execution")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        start_button = st.button(
            "▶️ Start Processing", 
            type="primary",
            disabled=(not url or st.session_state.process_status == "running")
        )
    
    with col2:
        if st.session_state.process_status == "running":
            st.button("⏸️ Processing...", disabled=True)
        else:
            stop_button = st.button("⏹️ Stop", disabled=True) # Placeholder, full stop logic is complex
    
    # Progress display area
    progress_placeholder = st.empty()
    status_messages_area = st.expander("📋 Detailed Log", expanded=False)
    
    if start_button and url and st.session_state.process_status != "running":
        st.session_state.process_status = "running"
        st.session_state.progress_messages = []
        st.session_state.results = {}

        # Use a placeholder for the progress bar and text
        progress_bar = progress_placeholder.progress(0)
        progress_text = progress_placeholder.text("Starting...")
        
        # Thread for background processing
        # Keep a reference to the thread to manage it if needed (e.g., for stopping)
        # current_thread = threading.Thread(
        #     target=run_processing_thread, 
        #     args=(url, config, progress_bar, progress_text, status_messages_area)
        # )
        # current_thread.start()
        # For Streamlit, direct threading can be tricky with state. Consider asyncio or background tasks pattern.
        
        # Simplified direct call for now, will block UI but easier for initial setup
        # For a better UX, use threading or asyncio as sketched above.
        # Re-enable spinner for visual feedback during blocking operations
        with st.spinner("Processing your request... Please wait."):
            run_processing_thread(url, config, progress_bar, progress_text, status_messages_area)

        # After processing completes (or if using threading, this would be handled differently)
        if st.session_state.process_status == "finished":
            progress_text.text("✅ Process completed!")
            progress_bar.progress(100)
            # Explicitly call display results here if not using a reactive flow
            # create_results_display_section() # Handled by main loop reacting to state change

def run_processing_thread(url, config, progress_bar, progress_text, status_messages_area):
    """Target function for the processing thread or direct call"""
    # Local progress tracking for the bar (0 to 100)
    current_progress_value = 0

    def update_progress(message, increment=None):
        nonlocal current_progress_value
        st.session_state.progress_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        with status_messages_area:
            # Display messages in reverse chronological order (newest first)
            for msg in reversed(st.session_state.progress_messages[-20:]): # Show last 20 messages
                st.text(msg)
        
        progress_text.text(message)
        if increment is not None and current_progress_value < 100:
            current_progress_value = min(100, current_progress_value + increment)
        progress_bar.progress(current_progress_value)

    try:
        results = st.session_state.workflow_manager.process_url(url, config, update_progress)
        st.session_state.results = results
        update_progress("✅ Workflow finished.", 100) # Ensure 100% at the end
    except Exception as e:
        update_progress(f"❌ Critical error in workflow: {e}", 100)
        st.session_state.results['errors'] = st.session_state.results.get('errors', []) + [f"Critical error: {e}"]
    finally:
        st.session_state.process_status = "finished"
        # Ensure UI updates after thread finishes if it were a real thread
        # st.experimental_rerun() # Use with caution, can cause loops.
        # Better to rely on Streamlit's natural rerun from widget interactions or state changes.

def create_results_display_section():
    """Display results of downloads and transcriptions"""
    if st.session_state.process_status == "finished" and st.session_state.results:
        st.header("📊 Results")
        results = st.session_state.results
        
        # Display errors first
        if results.get('errors'):
            st.subheader("⚠️ Errors & Warnings")
            for error_msg in results['errors']:
                st.error(error_msg)
        
        # Display downloads
        if results.get('downloads'):
            st.subheader("📥 Downloaded Files")
            # Create a DataFrame for better display
            download_data = []
            for file_path_str in results['downloads']:
                file_path = Path(file_path_str)
                try:
                    file_size = file_path.stat().st_size / (1024 * 1024) # MB
                    download_data.append({
                        "File Name": file_path.name,
                        "Size (MB)": f"{file_size:.2f}",
                        "Path": str(file_path) 
                        # "Download": f'<a href="data:application/octet-stream;base64,{...}" download="{file_path.name}">Download</a>'
                    })
                    # Provide download buttons (requires file to be accessible by Streamlit server)
                    with open(file_path, "rb") as fp:
                        st.download_button(
                            label=f"Download {file_path.name}",
                            data=fp,
                            file_name=file_path.name,
                            mime= f"audio/mpeg" if file_path.suffix == ".mp3" else "video/mp4" if file_path.suffix == ".mp4" else "application/octet-stream"
                        )
                except FileNotFoundError:
                     download_data.append({"File Name": file_path.name, "Size (MB)": "N/A", "Path": "File not found after processing (possibly cleaned up)"})
                except Exception as e:
                    download_data.append({"File Name": file_path.name, "Size (MB)": "N/A", "Path": f"Error accessing: {e}"})

            if download_data:
                df_downloads = pd.DataFrame(download_data)
                st.dataframe(df_downloads, use_container_width=True, hide_index=True)
        
        # Display transcriptions
        if results.get('transcriptions'):
            st.subheader("📝 Transcribed Files")
            transcription_data = []
            for file_path_str in results['transcriptions']:
                file_path = Path(file_path_str)
                try:
                    file_size = file_path.stat().st_size / 1024 # KB
                    transcription_data.append({
                        "File Name": file_path.name,
                        "Size (KB)": f"{file_size:.2f}",
                        "Path": str(file_path)
                    })
                     # Provide download buttons for transcripts
                    with open(file_path, "rb") as fp:
                        st.download_button(
                            label=f"Download {file_path.name}",
                            data=fp,
                            file_name=file_path.name,
                            mime="text/plain" if file_path.suffix == ".txt" else "application/json" if file_path.suffix == ".json" else "application/octet-stream"
                        )
                except FileNotFoundError:
                     transcription_data.append({"File Name": file_path.name, "Size (KB)": "N/A", "Path": "File not found after processing"})
                except Exception as e:
                    transcription_data.append({"File Name": file_path.name, "Size (KB)": "N/A", "Path": f"Error accessing: {e}"})
           
            if transcription_data:
                df_transcriptions = pd.DataFrame(transcription_data)
                st.dataframe(df_transcriptions, use_container_width=True, hide_index=True)
            
            # Option to download all as zip
            # Consider zipping all results (downloads + transcriptions) into one file.
            # This is more complex due to multiple file types and ensuring they exist.
            # For now, individual downloads are provided.

            # Create a zip of all result files (downloads and transcriptions)
            # This should be done carefully, ensuring files exist at zipping time
            all_files_to_zip = results.get('downloads', []) + results.get('transcriptions', [])
            valid_files_to_zip = [Path(f) for f in all_files_to_zip if Path(f).exists()]

            if valid_files_to_zip:
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for file_path_obj in valid_files_to_zip:
                        # Add file to zip, using a relative path within the zip
                        # e.g., downloads/video.mp4 or transcriptions/video/transcript.txt
                        # This depends on how paths are stored and what structure is desired in the zip.
                        # For simplicity, just adding with their names for now.
                        arcname = f"{file_path_obj.parent.name}/{file_path_obj.name}" if file_path_obj.parent.name in ["youtube_downloads", "transcriptions"] else file_path_obj.name
                        if file_path_obj.parent.parent.name == "transcriptions": # e.g. transcriptions/video_stem/file.txt
                             arcname = f"transcriptions/{file_path_obj.parent.name}/{file_path_obj.name}"
                        elif file_path_obj.parent.name == "youtube_downloads":
                             arcname = f"downloads/{file_path_obj.name}"
                        else: # Fallback for unexpected structure
                             arcname = file_path_obj.name
                        
                        zf.write(file_path_obj, arcname=arcname)
                
                zip_buffer.seek(0)
                st.download_button(
                    label="📦 Download All Results as ZIP",
                    data=zip_buffer,
                    file_name="youtube_downloader_results.zip",
                    mime="application/zip"
                )

        if not results.get('downloads') and not results.get('transcriptions') and not results.get('errors'):
            st.info("No results to display. Process may not have run or yielded no output.")

# --- Main Application Logic ---
def main():
    """Main function to run the Streamlit app"""
    st.markdown("<div class='main-header'><h1>🎥 YouTube Downloader & Transcriber <sup>v2</sup></h1></div>", unsafe_allow_html=True)
    
    initialize_session_state()
    config = create_sidebar_config()
    url, url_type = create_url_input_section()
    
    # Example Usage Section (Collapsible)
    with st.expander("✨ Example URLs & Quick Start", expanded=False):
        st.markdown("""
        Quickly test with these examples:
        - **Single Video:** `https://www.youtube.com/watch?v=dQw4w9WgXcQ`
        - **Playlist:** `https://www.youtube.com/playlist?list=PLMC9KNkIncKtPzgY-5rmFqZFvl5aiHr_J` (Short example playlist)
        - **Channel:** `https://www.youtube.com/@LinusTechTips/videos` (Fetches recent videos from uploads tab)
        
        **Instructions:**
        1. Paste a YouTube URL above.
        2. Click "Validate URL".
        3. Configure settings in the sidebar (download format, transcription options, etc.).
        4. Click "Start Processing".
        5. Results will appear below.
        """)
        col1, col2, col3 = st.columns(3)
        if col1.button("Use Video Example", key="vid_ex"):
            st.session_state.example_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            st.rerun()
        if col2.button("Use Playlist Example", key="play_ex"):
            st.session_state.example_url = "https://www.youtube.com/playlist?list=PLMC9KNkIncKtPzgY-5rmFqZFvl5aiHr_J"
            st.rerun()
        if col3.button("Use Channel Example", key="chan_ex"):
            st.session_state.example_url = "https://www.youtube.com/@LinusTechTips/videos"
            st.rerun()

    if url and url_type != "invalid":
        create_process_execution_section(url, url_type, config)
    
    create_results_display_section()

if __name__ == "__main__":
    main() 