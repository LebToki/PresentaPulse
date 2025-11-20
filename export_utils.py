"""
Export utilities for custom resolution, GIF, and frame sequence export
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
    logging.warning("OpenCV not available for export processing")

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


def export_custom_resolution(video_path: str, output_path: str, width: int, height: int,
                            maintain_aspect: bool = True, quality: str = 'high') -> bool:
    """
    Export video at custom resolution.
    
    Args:
        video_path: Input video path
        output_path: Output video path
        width: Target width
        height: Target height
        maintain_aspect: Maintain aspect ratio (adds padding if True)
        quality: Quality preset ('low', 'medium', 'high', 'ultra')
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Quality presets for encoding
        quality_settings = {
            'low': {'crf': '28', 'preset': 'fast'},
            'medium': {'crf': '23', 'preset': 'medium'},
            'high': {'crf': '18', 'preset': 'slow'},
            'ultra': {'crf': '15', 'preset': 'veryslow'}
        }
        
        settings = quality_settings.get(quality, quality_settings['high'])
        
        if maintain_aspect:
            # Scale with padding to maintain aspect ratio
            filter_complex = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"
        else:
            # Stretch to exact dimensions
            filter_complex = f"scale={width}:{height}"
        
        command = [
            FFMPEG_PATH,
            '-i', str(video_path),
            '-vf', filter_complex,
            '-c:v', 'libx264',
            '-crf', settings['crf'],
            '-preset', settings['preset'],
            '-pix_fmt', 'yuv420p',
            '-y',
            str(output_path)
        ]
        
        subprocess.run(command, capture_output=True, check=True)
        
        if os.path.exists(output_path):
            logging.info(f"Custom resolution export successful: {output_path}")
            return True
        else:
            logging.error("Custom resolution export failed - output file not created")
            return False
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Error exporting custom resolution: {e.stderr.decode() if e.stderr else str(e)}")
        return False
    except Exception as e:
        logging.error(f"Error in export_custom_resolution: {str(e)}")
        return False


def export_gif(video_path: str, output_path: str, fps: int = 15, 
              width: int = 512, optimize: bool = True, 
              colors: int = 256) -> bool:
    """
    Export video as animated GIF.
    
    Args:
        video_path: Input video path
        output_path: Output GIF path
        fps: Frame rate for GIF (lower = smaller file)
        width: Width of GIF (height auto-calculated)
        optimize: Use GIF optimization
        colors: Number of colors (256 max, lower = smaller file)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # First, create palette for better quality
        palette_path = str(output_path).replace('.gif', '_palette.png')
        
        # Generate palette
        palette_command = [
            FFMPEG_PATH,
            '-i', str(video_path),
            '-vf', f'fps={fps},scale={width}:-1:flags=lanczos,palettegen=max_colors={colors}',
            '-y',
            palette_path
        ]
        
        subprocess.run(palette_command, capture_output=True, check=True)
        
        # Create GIF using palette
        gif_command = [
            FFMPEG_PATH,
            '-i', str(video_path),
            '-i', palette_path,
            '-filter_complex', f'fps={fps},scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse',
            '-y',
            str(output_path)
        ]
        
        subprocess.run(gif_command, capture_output=True, check=True)
        
        # Cleanup palette file
        if os.path.exists(palette_path):
            os.remove(palette_path)
        
        if os.path.exists(output_path):
            logging.info(f"GIF export successful: {output_path}")
            return True
        else:
            logging.error("GIF export failed - output file not created")
            return False
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Error exporting GIF: {e.stderr.decode() if e.stderr else str(e)}")
        return False
    except Exception as e:
        logging.error(f"Error in export_gif: {str(e)}")
        return False


def export_frame_sequence(video_path: str, output_dir: str, 
                         format: str = 'png', prefix: str = 'frame',
                         start_frame: int = 0, end_frame: Optional[int] = None,
                         step: int = 1) -> Tuple[bool, int]:
    """
    Export video frames as image sequence.
    
    Args:
        video_path: Input video path
        output_dir: Output directory for frames
        format: Image format ('png', 'jpg', 'webp')
        prefix: Filename prefix
        start_frame: Start frame number (0-indexed)
        end_frame: End frame number (None = all frames)
        step: Frame step (1 = all frames, 2 = every other frame, etc.)
    
    Returns:
        Tuple of (success, frame_count)
    """
    try:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        # Build ffmpeg command
        command = [
            FFMPEG_PATH,
            '-i', str(video_path),
        ]
        
        # Add frame selection filters
        filters = []
        if start_frame > 0:
            filters.append(f'select=gte(n\\,{start_frame})')
        if end_frame is not None:
            filters.append(f'select=lte(n\\,{end_frame})')
        if step > 1:
            filters.append(f'select=not(mod(n\\,{step}))')
        
        if filters:
            filter_complex = ','.join(filters)
            command.extend(['-vf', filter_complex])
        
        # Output pattern
        output_pattern = str(output_dir_path / f'{prefix}_%06d.{format}')
        command.extend(['-y', output_pattern])
        
        subprocess.run(command, capture_output=True, check=True)
        
        # Count exported frames
        frame_count = len(list(output_dir_path.glob(f'{prefix}_*.{format}')))
        
        if frame_count > 0:
            logging.info(f"Frame sequence export successful: {frame_count} frames to {output_dir}")
            return True, frame_count
        else:
            logging.error("Frame sequence export failed - no frames exported")
            return False, 0
    
    except subprocess.CalledProcessError as e:
        logging.error(f"Error exporting frame sequence: {e.stderr.decode() if e.stderr else str(e)}")
        return False, 0
    except Exception as e:
        logging.error(f"Error in export_frame_sequence: {str(e)}")
        return False, 0


def get_video_resolution(video_path: str) -> Tuple[int, int]:
    """Get video resolution (width, height)."""
    try:
        import subprocess
        import json
        
        command = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            str(video_path)
        ]
        
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        if 'streams' in data and len(data['streams']) > 0:
            width = data['streams'][0].get('width', 0)
            height = data['streams'][0].get('height', 0)
            return width, height
        
        return 0, 0
    except Exception:
        return 0, 0

