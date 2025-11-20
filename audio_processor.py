"""
Audio processing utilities for video enhancement
Extract, sync, and enhance audio for animated videos
"""
import subprocess
import logging
import os
from pathlib import Path
from typing import Optional, Tuple
import tempfile

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available for audio processing")

def find_ffmpeg():
    """Find ffmpeg executable."""
    import shutil
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    import platform
    if platform.system() == 'Windows':
        common_paths = [
            'C:\\ffmpeg\\bin\\ffmpeg.exe',
            'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe',
        ]
    else:
        common_paths = [
            '/usr/bin/ffmpeg',
            '/usr/local/bin/ffmpeg',
        ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return 'ffmpeg'

FFMPEG_PATH = find_ffmpeg()

def find_ffprobe():
    """Find ffprobe executable."""
    import shutil
    ffprobe_path = shutil.which('ffprobe')
    if ffprobe_path:
        return ffprobe_path
    
    import platform
    if platform.system() == 'Windows':
        common_paths = [
            'C:\\ffmpeg\\bin\\ffprobe.exe',
            'C:\\Program Files\\ffmpeg\\bin\\ffprobe.exe',
        ]
    else:
        common_paths = [
            '/usr/bin/ffprobe',
            '/usr/local/bin/ffprobe',
        ]
    
    for path in common_paths:
        if os.path.exists(path):
            return path
    
    return 'ffprobe'

FFPROBE_PATH = find_ffprobe()


def extract_audio(video_path: str, output_audio_path: Optional[str] = None) -> Optional[str]:
    """
    Extract audio from video file.
    
    Args:
        video_path: Path to video file
        output_audio_path: Optional output path (defaults to temp file)
    
    Returns:
        Path to extracted audio file, or None if extraction failed
    """
    try:
        if not os.path.exists(video_path):
            logging.error(f"Video file not found: {video_path}")
            return None
        
        # Create output path if not provided
        if output_audio_path is None:
            temp_dir = tempfile.gettempdir()
            output_audio_path = os.path.join(temp_dir, f"extracted_audio_{os.path.basename(video_path)}.wav")
        
        # Extract audio using ffmpeg
        command = [
            FFMPEG_PATH,
            '-i', str(video_path),
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit
            '-ar', '44100',  # Sample rate
            '-ac', '2',  # Stereo
            '-y',  # Overwrite
            str(output_audio_path)
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        
        if os.path.exists(output_audio_path):
            logging.info(f"Audio extracted to: {output_audio_path}")
            return output_audio_path
        else:
            logging.error("Audio extraction failed - output file not created")
            return None
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting audio: {e.stderr}")
        return None
    except Exception as e:
        logging.error(f"Error in extract_audio: {str(e)}")
        return None


def get_audio_duration(audio_path: str) -> float:
    """Get duration of audio file in seconds."""
    try:
        command = [
            FFPROBE_PATH,
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(audio_path)
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logging.warning(f"Could not get audio duration: {e}")
        return 0.0


def get_video_duration(video_path: str) -> float:
    """Get duration of video file in seconds."""
    try:
        command = [
            FFPROBE_PATH,
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logging.warning(f"Could not get video duration: {e}")
        return 0.0


def sync_audio_to_video(audio_path: str, video_path: str, output_path: str,
                       loop_audio: bool = False, normalize: bool = False) -> bool:
    """
    Sync audio to video, matching durations.
    
    Args:
        audio_path: Path to audio file
        video_path: Path to video file
        output_path: Output video path with audio
        loop_audio: Loop audio if shorter than video
        normalize: Normalize audio levels
    
    Returns:
        True if successful, False otherwise
    """
    try:
        audio_duration = get_audio_duration(audio_path)
        video_duration = get_video_duration(video_path)
        
        # Build ffmpeg command
        command = [
            FFMPEG_PATH,
            '-i', str(video_path),
            '-i', str(audio_path),
            '-c:v', 'copy',  # Copy video stream (no re-encode)
        ]
        
        # Audio filter for normalization
        if normalize:
            command.extend(['-af', 'loudnorm=I=-16:TP=-1.5:LRA=11'])
        
        # Handle audio duration mismatch
        if audio_duration < video_duration and loop_audio:
            # Loop audio to match video length
            command.extend(['-filter_complex', f'[1:0]aloop=loop=-1:size=2e+09[a]'])
            command.extend(['-map', '0:v:0', '-map', '[a]'])
        elif audio_duration > video_duration:
            # Trim audio to match video
            command.extend(['-shortest'])
        else:
            # Use audio as-is
            command.extend(['-map', '0:v:0', '-map', '1:a:0'])
        
        command.extend([
            '-c:a', 'aac',
            '-b:a', '192k',
            '-strict', 'experimental',
            '-y',
            str(output_path)
        ])
        
        subprocess.run(command, capture_output=True, check=True)
        
        if os.path.exists(output_path):
            logging.info(f"Audio synced to video: {output_path}")
            return True
        else:
            logging.error("Audio sync failed - output file not created")
            return False
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Error syncing audio: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Error in sync_audio_to_video: {str(e)}")
        return False


def add_background_music(video_path: str, music_path: str, output_path: str,
                        music_volume: float = 0.3, original_audio_volume: float = 1.0,
                        normalize: bool = True) -> bool:
    """
    Add background music to video with volume mixing.
    
    Args:
        video_path: Path to video file
        music_path: Path to background music file
        output_path: Output video path
        music_volume: Volume of background music (0.0-1.0)
        original_audio_volume: Volume of original audio (0.0-1.0)
        normalize: Normalize audio levels
    
    Returns:
        True if successful, False otherwise
    """
    try:
        if not os.path.exists(music_path):
            logging.error(f"Music file not found: {music_path}")
            return False
        
        video_duration = get_video_duration(video_path)
        music_duration = get_audio_duration(music_path)
        
        # Build audio filter
        audio_filters = []
        
        # Check if video has audio
        has_audio = check_video_has_audio(video_path)
        
        if has_audio:
            # Mix original audio with background music
            if music_duration < video_duration:
                # Loop music
                audio_filters.append(f'[1:a]aloop=loop=-1:size=2e+09,volume={music_volume}[music]')
                audio_filters.append(f'[0:a]volume={original_audio_volume}[orig]')
                audio_filters.append(f'[orig][music]amix=inputs=2:duration=first:dropout_transition=2[mixed]')
            else:
                # Trim music to video length
                audio_filters.append(f'[1:a]atrim=0:{video_duration},volume={music_volume}[music]')
                audio_filters.append(f'[0:a]volume={original_audio_volume}[orig]')
                audio_filters.append(f'[orig][music]amix=inputs=2:duration=first:dropout_transition=2[mixed]')
        else:
            # Just add music (no original audio)
            if music_duration < video_duration:
                audio_filters.append(f'[1:a]aloop=loop=-1:size=2e+09,volume={music_volume}[mixed]')
            else:
                audio_filters.append(f'[1:a]atrim=0:{video_duration},volume={music_volume}[mixed]')
        
        # Add normalization if requested
        if normalize:
            audio_filters.append('[mixed]loudnorm=I=-16:TP=-1.5:LRA=11[final]')
            output_label = '[final]'
        else:
            output_label = '[mixed]'
        
        filter_complex = ';'.join(audio_filters)
        
        command = [
            FFMPEG_PATH,
            '-i', str(video_path),
            '-i', str(music_path),
            '-filter_complex', filter_complex,
            '-map', '0:v:0',
            '-map', output_label,
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            '-y',
            str(output_path)
        ]
        
        subprocess.run(command, capture_output=True, check=True)
        
        if os.path.exists(output_path):
            logging.info(f"Background music added: {output_path}")
            return True
        else:
            logging.error("Adding background music failed - output file not created")
            return False
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Error adding background music: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Error in add_background_music: {str(e)}")
        return False


def normalize_audio(audio_path: str, output_path: str) -> bool:
    """
    Normalize audio levels using loudnorm filter.
    
    Args:
        audio_path: Path to input audio file
        output_path: Path to output normalized audio file
    
    Returns:
        True if successful, False otherwise
    """
    try:
        command = [
            FFMPEG_PATH,
            '-i', str(audio_path),
            '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11',
            '-y',
            str(output_path)
        ]
        
        subprocess.run(command, capture_output=True, check=True)
        
        if os.path.exists(output_path):
            logging.info(f"Audio normalized: {output_path}")
            return True
        else:
            logging.error("Audio normalization failed")
            return False
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Error normalizing audio: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Error in normalize_audio: {str(e)}")
        return False


def check_video_has_audio(video_path: str) -> bool:
    """Check if video file has audio stream."""
    try:
        command = [
            FFPROBE_PATH,
            '-v', 'error',
            '-select_streams', 'a:0',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        return 'audio' in result.stdout.lower()
    except Exception:
        return False


def process_audio_for_video(video_path: str, source_video_path: Optional[str] = None,
                           background_music_path: Optional[str] = None,
                           music_volume: float = 0.3, normalize: bool = True,
                           loop_audio: bool = False, output_path: Optional[str] = None) -> Optional[str]:
    """
    Complete audio processing pipeline for video.
    
    Args:
        video_path: Path to video file (may or may not have audio)
        source_video_path: Path to source video to extract audio from
        background_music_path: Path to background music file
        music_volume: Volume of background music (0.0-1.0)
        normalize: Normalize audio levels
        loop_audio: Loop audio if shorter than video
        output_path: Output video path (defaults to input with _audio suffix)
    
    Returns:
        Path to output video with audio, or None if failed
    """
    try:
        if output_path is None:
            video_path_obj = Path(video_path)
            output_path = str(video_path_obj.parent / f"{video_path_obj.stem}_with_audio{video_path_obj.suffix}")
        
        # Step 1: Extract audio from source video if provided
        extracted_audio = None
        if source_video_path and os.path.exists(source_video_path):
            extracted_audio = extract_audio(source_video_path)
        
        # Step 2: Add background music if provided
        if background_music_path and os.path.exists(background_music_path):
            if extracted_audio:
                # First sync extracted audio
                temp_video = str(Path(output_path).parent / 'temp_with_audio.mp4')
                sync_audio_to_video(extracted_audio, video_path, temp_video, loop_audio, normalize)
                # Then add background music
                success = add_background_music(temp_video, background_music_path, output_path,
                                             music_volume, 1.0, normalize)
                # Cleanup temp file
                if os.path.exists(temp_video):
                    os.remove(temp_video)
                return output_path if success else None
            else:
                # Just add background music
                success = add_background_music(video_path, background_music_path, output_path,
                                             music_volume, 0.0, normalize)
                return output_path if success else None
        
        # Step 3: Sync extracted audio if no background music
        elif extracted_audio:
            success = sync_audio_to_video(extracted_audio, video_path, output_path, loop_audio, normalize)
            return output_path if success else None
        
        # No audio processing needed
        return video_path
    
    except Exception as e:
        logging.error(f"Error in process_audio_for_video: {str(e)}")
        return None

