"""
Video utility functions for PresentaPulse
This module provides basic video processing utilities.
"""
import sys
import os
from pathlib import Path

# Add project root to path to import from root-level modules
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import functions from video_enhanced.py
try:
    from video_enhanced import has_audio_stream, exec_cmd
except ImportError:
    # Fallback implementations if video_enhanced is not available
    import subprocess
    import logging
    import shutil
    
    def find_ffprobe():
        """Find ffprobe executable."""
        ffprobe_path = shutil.which('ffprobe')
        if ffprobe_path:
            return ffprobe_path
        return 'ffprobe'
    
    FFPROBE_PATH = find_ffprobe()
    
    def exec_cmd(cmd, total_steps=None, desc="Processing"):
        """Execute command with progress tracking."""
        try:
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                universal_newlines=True
            )
            result = process.communicate()[0]
            return result
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if hasattr(e, 'stderr') else str(e)
            logging.error(f"Command '{cmd}' failed with error: {error_msg}")
            raise
        except Exception as e:
            logging.error(f"Error executing command: {str(e)}")
            raise
    
    def has_audio_stream(video_path):
        """Check if video has audio stream."""
        cmd = [FFPROBE_PATH, '-v', 'error', '-select_streams', 'a', 
               '-show_entries', 'stream=codec_type', '-of',
               'default=noprint_wrappers=1:nokey=1', str(video_path)]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            return 'audio' in result.stdout
        except Exception as e:
            logging.warning(f"Error checking audio stream: {e}. Assuming no audio.")
            return False


# Basic VideoEnhancer class
import subprocess
from multiprocessing import Pool
from pathlib import Path


def enhance_frame(frame_path, esrgan_script_path, esrgan_model_path, esrgan_output_dir):
    """Enhance a single frame using Real-ESRGAN."""
    command = [
        'python', str(esrgan_script_path), '-n', 'RealESRGAN_x4plus_anime_6B',
        '-i', str(frame_path), '-o', str(esrgan_output_dir / frame_path.name),
        '--model_path', str(esrgan_model_path)
    ]
    subprocess.run(command, check=True)


class VideoEnhancer:
    """
    Basic video enhancer for frame-by-frame enhancement using Real-ESRGAN.
    
    This is a fallback implementation when the enhanced version is not available.
    """
    
    def __init__(self, esrgan_script_path, esrgan_model_path, esrgan_output_dir):
        """
        Initialize VideoEnhancer.
        
        Args:
            esrgan_script_path: Path to Real-ESRGAN inference script
            esrgan_model_path: Path to Real-ESRGAN model file
            esrgan_output_dir: Directory to save enhanced frames
        """
        self.esrgan_script_path = Path(esrgan_script_path)
        self.esrgan_model_path = Path(esrgan_model_path)
        self.esrgan_output_dir = Path(esrgan_output_dir)
        
        # Ensure output directory exists
        self.esrgan_output_dir.mkdir(parents=True, exist_ok=True)
    
    def enhance_frames(self, input_dir):
        """
        Enhance all frames in the input directory.
        
        Args:
            input_dir: Directory containing input frames (PNG files)
        """
        input_dir = Path(input_dir)
        frame_paths = list(input_dir.glob('*.png'))
        
        if not frame_paths:
            import logging
            logging.warning(f"No PNG frames found in {input_dir}")
            return
        
        # Enhance frames using multiprocessing
        with Pool() as pool:
            pool.starmap(
                enhance_frame,
                [(frame, self.esrgan_script_path, self.esrgan_model_path, self.esrgan_output_dir) 
                 for frame in frame_paths]
            )

