import whisper
import moviepy.editor as mp
from pathlib import Path
import json
import logging
import torch # For device check
import re
from datetime import timedelta

logger = logging.getLogger(__name__)

class VideoTranscriber:
    """Transcribes video or audio files using OpenAI's Whisper model."""

    def __init__(self, model_size: str = "base", device: str = "auto", output_dir: str = "transcriptions"):
        self.model_size = model_size
        self.device = self._get_device(device)
        self.output_dir = Path(output_dir)
        # Ensure the base transcription output directory exists upon initialization
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.model = whisper.load_model(self.model_size, device=self.device)
            logger.info(f"Whisper model '{self.model_size}' loaded onto {self.device}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model '{self.model_size}': {e}")
            # Potentially fallback to CPU or a smaller model if robust error handling is desired
            # For now, re-raise to indicate a critical issue with transcriber setup
            raise

    def _get_device(self, requested_device: str) -> str:
        """Determine computation device (CPU or CUDA if available)."""
        if requested_device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested_device

    def _extract_audio(self, video_path: Path) -> Path:
        """Extracts audio from video file and saves as MP3."""
        try:
            video_clip = mp.VideoFileClip(str(video_path))
            # Use video_path.stem to avoid issues with multiple dots in filename
            audio_filename = f"{video_path.stem}_audio.mp3"
            # Save audio in a temporary sub-directory or a dedicated audio cache dir
            # For now, let's place it alongside the video but with _audio suffix
            # to avoid filename clashes if original was already mp3.
            # Better: use a temp dir for intermediates.
            temp_audio_dir = self.output_dir / ".temp_audio"
            temp_audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = temp_audio_dir / audio_filename
            
            video_clip.audio.write_audiofile(str(audio_path), codec='mp3', logger=None) # moviepy logger is verbose
            video_clip.close() # Release file handle
            logger.info(f"Audio extracted to: {audio_path}")
            return audio_path
        except Exception as e:
            logger.error(f"Error extracting audio from {video_path.name}: {e}")
            # Close clip if it was opened and an error occurred during write_audiofile or after
            if 'video_clip' in locals() and hasattr(video_clip, 'close'):
                video_clip.close()
            raise

    def _format_timestamp(self, seconds: float, decimal_marker='.') -> str:
        """Formats seconds into HH:MM:SS,mmm or HH:MM:SS.mmm timestamp."""
        delta = timedelta(seconds=seconds)
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = delta.microseconds // 1000
        return f"{hours:02}:{minutes:02}:{seconds:02}{decimal_marker}{milliseconds:03}"

    def _create_srt_content(self, result: dict, timestamp_duration: int) -> str:
        """Creates SRT formatted transcript content from Whisper result."""
        srt_content = []
        segment_id = 1
        for segment in result.get('segments', []):
            start_time = self._format_timestamp(segment['start'], decimal_marker=',')
            end_time = self._format_timestamp(segment['end'], decimal_marker=',')
            text = segment['text'].strip()
            srt_content.append(f"{segment_id}\n{start_time} --> {end_time}\n{text}\n")
            segment_id += 1
        return "\n".join(srt_content)

    def _create_vtt_content(self, result: dict, timestamp_duration: int) -> str:
        """Creates VTT formatted transcript content from Whisper result."""
        vtt_content = ["WEBVTT\n"]
        for segment in result.get('segments', []):
            start_time = self._format_timestamp(segment['start'])
            end_time = self._format_timestamp(segment['end'])
            text = segment['text'].strip()
            vtt_content.append(f"{start_time} --> {end_time}\n{text}\n")
        return "\n".join(vtt_content)

    def _save_transcript(self, content: str, output_path: Path):
        """Saves transcript content to a file."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"Transcript saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving transcript to {output_path}: {e}")
            raise

    def transcribe(self, 
                   media_path_str: str, 
                   output_formats: List[str] = None, 
                   include_metadata: bool = True,
                   timestamp_duration: int = 8, # Not directly used by Whisper, but for SRT/VTT grouping if needed
                   video_metadata: dict = None, # Optional metadata from downloader
                   output_subpath: str = None) -> dict: # Subpath within self.output_dir
        """Transcribes a video or audio file.

        Args:
            media_path_str: Path to the media file.
            output_formats: List of formats like ['txt', 'json', 'srt', 'vtt'].
            include_metadata: Whether to include metadata in JSON output.
            timestamp_duration: Desired duration of segments for SRT/VTT (Whisper decides actual segments).
            video_metadata: Optional pre-fetched video metadata.
            output_subpath: Optional sub-directory name within the main output_dir for this specific transcription.

        Returns:
            A dictionary containing paths to transcript files and metadata, or None if failed.
        """
        media_path = Path(media_path_str)
        if not media_path.exists():
            logger.error(f"Media file not found: {media_path}")
            return None
        
        if output_formats is None:
            output_formats = ['txt'] # Default to text if not specified

        # Determine the specific output directory for this transcription
        current_output_base_dir = self.output_dir
        if output_subpath:
            current_output_base_dir = self.output_dir / output_subpath
        current_output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Use media_path.stem for naming to avoid issues with multiple dots in filename
        base_filename = media_path.stem
        
        # Handle audio extraction if it's a video file
        # Whisper can handle video files directly, but extracting audio first can be more reliable
        # or allow specific audio processing if needed.
        # For simplicity and to leverage Whisper's direct video handling (if preferred or more efficient):
        # We can pass the video_path directly to whisper if it's a video format it supports.
        # However, yt-dlp often downloads as .mp4 (video) even if audio_only was requested, 
        # then post-processes to .mp3. If we get an .mp4 here, it might be intended as audio.
        # Let's stick to extracting audio if it's not already a common audio format.
        
        audio_path_to_transcribe = media_path
        is_video_file = media_path.suffix.lower() in ['.mp4', '.mkv', '.mov', '.avi', '.webm']
        # Check if it's not already a primary audio format Whisper handles well
        is_primary_audio = media_path.suffix.lower() in ['.mp3', '.wav', '.m4a', '.ogg']
        
        temp_audio_file_to_delete = None
        if is_video_file and not is_primary_audio: # If it's a video container, extract audio
            try:
                logger.info(f"Extracting audio from video: {media_path.name}")
                audio_path_to_transcribe = self._extract_audio(media_path)
                temp_audio_file_to_delete = audio_path_to_transcribe
            except Exception as e:
                logger.error(f"Audio extraction failed for {media_path.name}: {e}. Attempting transcription with original file.")
                # Fallback to transcribing original file if audio extraction fails
                audio_path_to_transcribe = media_path
        elif not is_video_file and not is_primary_audio:
            # If it's not a video and not a common audio, it might be unsupported or an intermediate.
            # Whisper might still handle it. Log a warning.
            logger.warning(f"File {media_path.name} is not a standard video or audio format. Transcription might fail or be inaccurate.")

        try:
            logger.info(f"Starting transcription for: {audio_path_to_transcribe.name}")
            # word_timestamps=True can be very verbose for JSON if not specifically needed
            result = self.model.transcribe(str(audio_path_to_transcribe), fp16=False, word_timestamps=False)
            logger.info(f"Transcription successful for: {audio_path_to_transcribe.name}")
        except Exception as e:
            logger.error(f"Whisper transcription failed for {audio_path_to_transcribe.name}: {e}")
            if temp_audio_file_to_delete and temp_audio_file_to_delete.exists():
                 try: temp_audio_file_to_delete.unlink(); logger.info(f"Cleaned up temp audio: {temp_audio_file_to_delete.name}")
                 except OSError: logger.error(f"Error deleting temp audio {temp_audio_file_to_delete.name}")
            return None
        finally:
            # Clean up temporary audio file if one was created
            if temp_audio_file_to_delete and temp_audio_file_to_delete.exists():
                try: 
                    temp_audio_file_to_delete.unlink()
                    logger.info(f"Cleaned up temp audio: {temp_audio_file_to_delete.name}")
                except OSError as e:
                    logger.error(f"Error deleting temporary audio file {temp_audio_file_to_delete.name}: {e}")
        
        output_files = {}
        # Save requested formats
        if 'txt' in output_formats:
            txt_path = current_output_base_dir / f"{base_filename}.txt"
            self._save_transcript(result['text'], txt_path)
            output_files['txt_file'] = str(txt_path)

        if 'srt' in output_formats:
            srt_content = self._create_srt_content(result, timestamp_duration)
            srt_path = current_output_base_dir / f"{base_filename}.srt"
            self._save_transcript(srt_content, srt_path)
            output_files['srt_file'] = str(srt_path)

        if 'vtt' in output_formats:
            vtt_content = self._create_vtt_content(result, timestamp_duration)
            vtt_path = current_output_base_dir / f"{base_filename}.vtt"
            self._save_transcript(vtt_content, vtt_path)
            output_files['vtt_file'] = str(vtt_path)
        
        # For JSON, include more details if requested
        if 'json' in output_formats:
            json_path = current_output_base_dir / f"{base_filename}_transcript.json"
            json_output = {
                'text': result['text'],
                'language': result['language'],
                'segments': result['segments'] # Segments are usually desired for JSON
            }
            if include_metadata and video_metadata:
                json_output['video_metadata'] = video_metadata
            
            try:
                json_path.parent.mkdir(parents=True, exist_ok=True)
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_output, f, indent=4, ensure_ascii=False)
                logger.info(f"JSON transcript saved to: {json_path}")
                output_files['json_file'] = str(json_path)
            except Exception as e:
                logger.error(f"Error saving JSON transcript to {json_path}: {e}")

        # Return dictionary of paths and potentially the raw result or metadata
        final_result_package = {
            **output_files,
            "language_detected": result.get('language')
        }
        if include_metadata and video_metadata:
            final_result_package["metadata"] = video_metadata
        
        return final_result_package

# Example Usage (for testing module directly)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # --- Configuration ---
    # Create a dummy video file for testing (e.g., a short MP4 or use an existing one)
    # Ensure FFmpeg is installed and accessible if testing with video files.
    TEST_MEDIA_DIR = Path.home() / "Downloads" / "TestWhisperMedia"
    TEST_MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy mp4 for testing if it doesn't exist
    dummy_video_path = TEST_MEDIA_DIR / "dummy_video.mp4"
    if not dummy_video_path.exists():
        try:
            # Create a very short, silent mp4 file using moviepy
            silence = mp.AudioClip(lambda t: [0,0], duration=2, fps=44100) # 2 sec stereo silence
            clip = mp.ImageClip(str(Path(whisper.__file__).parent / "assets" / "logo.png")) # Use whisper logo if available
            if not clip.img: # Fallback if logo not found or not an image
                import numpy as np
                clip = mp.ImageClip(np.zeros((100,100,3), dtype=np.uint8), duration=2)
            else:
                clip = clip.set_duration(2)
            
            clip = clip.set_audio(silence)
            clip.write_videofile(str(dummy_video_path), fps=1, codec="libx264", audio_codec="aac", logger=None)
            logger.info(f"Created dummy video: {dummy_video_path}")
        except Exception as e:
            logger.error(f"Could not create dummy video file for testing: {e}. Please create one manually.")
            logger.error("Skipping VideoTranscriber tests if dummy video is missing.")

    TRANSCRIPTION_OUTPUT_DIR = Path.home() / "Downloads" / "TestWhisperTranscripts"
    TRANSCRIPTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize transcriber
    # Use a small model for quick testing
    try:
        transcriber = VideoTranscriber(model_size="tiny", device="auto", output_dir=str(TRANSCRIPTION_OUTPUT_DIR))
        
        if dummy_video_path.exists():
            logger.info(f"\n--- Testing Transcription for: {dummy_video_path.name} ---")
            # Test with a subpath for this specific transcription
            transcription_result = transcriber.transcribe(
                str(dummy_video_path), 
                output_formats=['txt', 'json', 'srt', 'vtt'],
                include_metadata=True,
                video_metadata={"title": "Dummy Video Test", "id": "test001"},
                output_subpath=dummy_video_path.stem # Save in output_dir / dummy_video /
            )

            if transcription_result:
                logger.info("Transcription successful. Files created:")
                for key, path in transcription_result.items():
                    if "_file" in key and path: # Check if it's a file path
                        logger.info(f"  {key}: {path}")
            else:
                logger.error("Transcription failed.")
        else:
            logger.warning(f"Skipping transcription test as dummy video {dummy_video_path} was not found/created.")

    except Exception as e:
        logger.error(f"Failed to initialize or run VideoTranscriber test: {e}")

    logger.info(f"\n--- VideoTranscriber tests complete ---")
    logger.info(f"Check output directory for transcripts: {TRANSCRIPTION_OUTPUT_DIR}") 