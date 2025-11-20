"""
Enhanced video processing utilities with progress tracking and better error handling
"""
import subprocess
import logging
import shutil
import os
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Find ffprobe in PATH or use default location
def find_ffprobe():
    """Find ffprobe executable in system PATH or common locations."""
    # First try to find in PATH
    ffprobe_path = shutil.which('ffprobe')
    if ffprobe_path:
        return ffprobe_path
    
    # Try common installation locations
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
    
    # Fallback to just 'ffprobe' and hope it's in PATH
    return 'ffprobe'

FFPROBE_PATH = find_ffprobe()

def find_ffmpeg():
    """Find ffmpeg executable in system PATH or common locations."""
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


def exec_cmd(cmd, total_steps=None, desc="Processing"):
    """Execute command with progress tracking."""
    try:
        with tqdm(total=total_steps, desc=desc, unit="step", ncols=80) as pbar:
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                     universal_newlines=True, bufsize=1)
            while True:
                output = process.stdout.readline()
                if process.poll() is not None and output == '':
                    break
                if output:
                    pbar.update(1)
                    logging.debug(output.strip())
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
    cmd = [FFPROBE_PATH, '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=codec_type', '-of',
           'default=noprint_wrappers=1:nokey=1', str(video_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        return 'audio' in result.stdout
    except Exception as e:
        logging.warning(f"Error checking audio stream: {e}. Assuming no audio.")
        return False


def get_video_info(video_path):
    """Get video information (fps, duration, resolution)."""
    try:
        cmd = [
            FFPROBE_PATH, '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,r_frame_rate,duration',
            '-of', 'json', str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        info = json.loads(result.stdout)
        if 'streams' in info and len(info['streams']) > 0:
            stream = info['streams'][0]
            fps_str = stream.get('r_frame_rate', '30/1')
            fps = eval(fps_str) if '/' in fps_str else float(fps_str)
            return {
                'width': stream.get('width', 512),
                'height': stream.get('height', 512),
                'fps': fps,
                'duration': float(stream.get('duration', 0))
            }
    except Exception as e:
        logging.warning(f"Error getting video info: {e}. Using defaults.")
    return {'width': 512, 'height': 512, 'fps': 30, 'duration': 0}


def downscale_video(input_video_path, output_video_path, width=1280, progress_callback=None):
    """Downscale video with progress tracking."""
    try:
        if progress_callback:
            progress_callback(0.1, "Downscaling video...")
        
        command = [
            FFMPEG_PATH, '-i', str(input_video_path), 
            '-vf', f'scale={width}:-1',
            '-y',  # Overwrite output file
            str(output_video_path)
        ]
        subprocess.run(command, check=True, capture_output=True)
        
        if progress_callback:
            progress_callback(0.3, "Video downscaled")
        
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error downscaling video: {e.stderr.decode() if e.stderr else str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error in downscale_video: {str(e)}")
        raise


def extract_frames(video_path, output_dir, progress_callback=None):
    """Extract frames from video with progress tracking."""
    try:
        if progress_callback:
            progress_callback(0.3, "Extracting frames...")
        
        # Get frame count for progress
        info = get_video_info(video_path)
        frame_count = int(info['fps'] * info['duration']) if info['duration'] > 0 else 100
        
        command = [
            FFMPEG_PATH, '-i', str(video_path),
            '-vf', 'fps=30',  # Extract at 30fps
            '-y',
            str(output_dir / 'frame_%04d.png')
        ]
        subprocess.run(command, check=True, capture_output=True)
        
        if progress_callback:
            progress_callback(0.5, f"Extracted frames to {output_dir}")
        
        return frame_count
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting frames: {e.stderr.decode() if e.stderr else str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error in extract_frames: {str(e)}")
        raise


def apply_smoothing_filters(video_path, smoothing_strength=0.5, denoise_strength=0.3, 
                           stabilize=False, progress_callback=None):
    """Apply smoothing and denoising filters to video."""
    try:
        if progress_callback:
            progress_callback(0.95, "Applying smoothing filters...")
        
        # Create temporary smoothed video
        temp_path = str(video_path).replace('.mp4', '_smoothed_temp.mp4')
        
        filters = []
        
        # Temporal smoothing (minterpolate) - reduces jitter between frames
        if smoothing_strength > 0:
            # minterpolate: motion interpolation for smoother motion
            # mi_mode: mci (motion compensated interpolation)
            # mc: motion compensation
            # vsbmc: variable-size block motion compensation
            smoothing_value = int(smoothing_strength * 10)  # 0-10 scale
            filters.append(f"minterpolate=fps={smoothing_value*2+30}:mi_mode=mci:mc=aobmc:vsbmc=1")
        
        # Denoising (hqdn3d) - removes noise
        if denoise_strength > 0:
            # hqdn3d: high quality 3D denoise filter
            # luma_spatial, chroma_spatial, luma_temporal, chroma_temporal
            denoise_luma = denoise_strength * 4.0
            denoise_chroma = denoise_strength * 3.0
            filters.append(f"hqdn3d={denoise_luma}:{denoise_chroma}:{denoise_luma*0.8}:{denoise_chroma*0.8}")
        
        # Motion stabilization (deshake) - reduces camera shake
        if stabilize:
            filters.append("deshake")
        
        # Build filter string
        filter_string = ','.join(filters) if filters else None
        
        command = [FFMPEG_PATH, '-i', str(video_path), '-y']
        
        if filter_string:
            command.extend(['-vf', filter_string])
        
        command.extend([
            '-c:v', 'libx264',
            '-crf', '18',
            '-preset', 'medium',
            '-pix_fmt', 'yuv420p',
            temp_path
        ])
        
        subprocess.run(command, check=True, capture_output=True)
        
        # Replace original with smoothed version
        import shutil
        shutil.move(temp_path, str(video_path))
        
        if progress_callback:
            progress_callback(1.0, "Smoothing applied successfully!")
        
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error applying smoothing: {e.stderr.decode() if e.stderr else str(e)}")
        # Don't fail if smoothing fails, just log warning
        logging.warning("Smoothing failed, continuing without smoothing")
        return False
    except Exception as e:
        logging.warning(f"Error in apply_smoothing_filters: {str(e)}")
        return False


def reassemble_video(input_dir, output_video_path, fps=30, quality='high', 
                    format='mp4', audio_path=None, smoothing_strength=0.0,
                    denoise_strength=0.0, stabilize=False, progress_callback=None):
    """Reassemble frames into video with quality and format options."""
    try:
        if progress_callback:
            progress_callback(0.9, f"Reassembling video ({format.upper()})...")
        
        # Quality presets
        quality_settings = {
            'low': {'crf': '28', 'preset': 'fast'},
            'medium': {'crf': '23', 'preset': 'medium'},
            'high': {'crf': '18', 'preset': 'slow'},
            'ultra': {'crf': '15', 'preset': 'veryslow'}
        }
        
        settings = quality_settings.get(quality, quality_settings['high'])
        
        # Build video filters
        video_filters = []
        
        # Apply smoothing/denoising during encoding if requested
        if smoothing_strength > 0 or denoise_strength > 0 or stabilize:
            if smoothing_strength > 0:
                smoothing_value = int(smoothing_strength * 10)
                video_filters.append(f"minterpolate=fps={smoothing_value*2+30}:mi_mode=mci:mc=aobmc:vsbmc=1")
            
            if denoise_strength > 0:
                denoise_luma = denoise_strength * 4.0
                denoise_chroma = denoise_strength * 3.0
                video_filters.append(f"hqdn3d={denoise_luma}:{denoise_chroma}:{denoise_luma*0.8}:{denoise_chroma*0.8}")
            
            if stabilize:
                video_filters.append("deshake")
        
        command = [
            FFMPEG_PATH, '-framerate', str(fps), '-i',
            str(input_dir / 'frame_%04d.png'),
            '-c:v', 'libx264',
            '-crf', settings['crf'],
            '-preset', settings['preset'],
            '-pix_fmt', 'yuv420p',
            '-y'
        ]
        
        # Add video filters if any
        if video_filters:
            filter_string = ','.join(video_filters)
            # Insert filter before codec
            filter_idx = command.index('-c:v')
            command.insert(filter_idx, '-vf')
            command.insert(filter_idx + 1, filter_string)
        
        # Add audio if provided
        if audio_path and os.path.exists(audio_path):
            command.extend(['-i', str(audio_path), '-c:a', 'aac', '-strict', 'experimental'])
        
        # Format-specific options
        if format.lower() == 'webm':
            command[command.index('-c:v')] = '-c:v'
            command[command.index('libx264')] = 'libvpx-vp9'
        elif format.lower() == 'mov':
            command.extend(['-c:v', 'libx264', '-pix_fmt', 'yuv420p'])
        
        command.append(str(output_video_path))
        
        subprocess.run(command, check=True, capture_output=True)
        
        if progress_callback:
            progress_callback(1.0, "Video reassembled successfully!")
        
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error reassembling video: {e.stderr.decode() if e.stderr else str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error in reassemble_video: {str(e)}")
        raise


def enhance_frame(frame_path, esrgan_script_path, esrgan_model_name, esrgan_model_path, esrgan_output_dir):
    """Enhance a single frame using Real-ESRGAN."""
    try:
        command = [
            'python', str(esrgan_script_path), '-n', esrgan_model_name,
            '-i', str(frame_path), '-o', str(esrgan_output_dir / frame_path.name),
            '--model_path', str(esrgan_model_path)
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error enhancing frame {frame_path.name}: {e.stderr}")
        return False
    except Exception as e:
        logging.error(f"Error in enhance_frame: {str(e)}")
        return False


class VideoEnhancer:
    """Enhanced video enhancer with model selection and progress tracking."""
    
    # Available Real-ESRGAN models
    AVAILABLE_MODELS = {
        'RealESRGAN_x4plus': {
            'name': 'RealESRGAN_x4plus',
            'description': 'General purpose 4x upscaling',
            'file': 'RealESRGAN_x4plus.pth'
        },
        'RealESRGAN_x4plus_anime_6B': {
            'name': 'RealESRGAN_x4plus_anime_6B',
            'description': 'Anime/illustration optimized (6B parameters)',
            'file': 'RealESRGAN_x4plus_anime_6B.pth'
        },
        'realesr-animevideov3': {
            'name': 'realesr-animevideov3',
            'description': 'Anime video optimized',
            'file': 'realesr-animevideov3.pth'
        },
        'RealESRNet_x4plus': {
            'name': 'RealESRNet_x4plus',
            'description': 'Network-based 4x upscaling',
            'file': 'RealESRNet_x4plus.pth'
        }
    }
    
    def __init__(self, esrgan_script_path, pretrained_weights_dir, esrgan_output_dir):
        self.esrgan_script_path = esrgan_script_path
        self.pretrained_weights_dir = Path(pretrained_weights_dir)
        self.esrgan_output_dir = esrgan_output_dir
        
    def get_available_models(self):
        """Get list of available models based on files in pretrained_weights."""
        available = []
        for model_id, model_info in self.AVAILABLE_MODELS.items():
            model_path = self.pretrained_weights_dir / model_info['file']
            if model_path.exists():
                available.append({
                    'id': model_id,
                    'name': model_info['name'],
                    'description': model_info['description'],
                    'path': model_path
                })
        return available
    
    def enhance_frames(self, input_dir, model_name='RealESRGAN_x4plus_anime_6B', 
                      progress_callback=None):
        """Enhance frames using specified Real-ESRGAN model."""
        frame_paths = sorted(list(input_dir.glob('*.png')))
        
        if not frame_paths:
            logging.warning("No frames found to enhance")
            return []
        
        # Get model info
        model_info = self.AVAILABLE_MODELS.get(model_name, self.AVAILABLE_MODELS['RealESRGAN_x4plus_anime_6B'])
        model_path = self.pretrained_weights_dir / model_info['file']
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        total_frames = len(frame_paths)
        enhanced_frames = []
        
        if progress_callback:
            progress_callback(0.6, f"Enhancing {total_frames} frames with {model_info['name']}...")
        
        # Process frames with progress tracking
        with Pool() as pool:
            results = []
            for i, frame in enumerate(frame_paths):
                result = pool.apply_async(
                    enhance_frame,
                    (frame, self.esrgan_script_path, model_info['name'], model_path, self.esrgan_output_dir)
                )
                results.append((frame, result))
                
                if progress_callback and (i + 1) % max(1, total_frames // 10) == 0:
                    progress = 0.6 + (0.3 * (i + 1) / total_frames)
                    progress_callback(progress, f"Enhanced {i + 1}/{total_frames} frames...")
            
            # Wait for all results
            for frame, result in results:
                if result.get():
                    enhanced_frames.append(self.esrgan_output_dir / frame.name)
        
        if progress_callback:
            progress_callback(0.9, f"Enhanced {len(enhanced_frames)}/{total_frames} frames")
        
        return enhanced_frames

