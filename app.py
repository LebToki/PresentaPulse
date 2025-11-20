# coding: utf-8

import os
import subprocess
import logging
from pathlib import Path
import tyro
import gradio as gr
import os.path as osp
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV not available. Some features may be limited.")
from src.utils.helper import load_description
from src.gradio_pipeline import GradioPipeline
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.utils.video import has_audio_stream, exec_cmd, VideoEnhancer
try:
    from video_enhanced import (
        VideoEnhancer as EnhancedVideoEnhancer, 
        get_video_info, 
        downscale_video, 
        extract_frames, 
        reassemble_video,
        find_ffmpeg
    )
    ENHANCED_VIDEO_AVAILABLE = True
except ImportError:
    ENHANCED_VIDEO_AVAILABLE = False
    logging.warning("Enhanced video processing not available, using basic version")
    find_ffmpeg = lambda: 'ffmpeg'

try:
    from face_detection import FaceDetector, detect_faces_in_image
    FACE_DETECTION_AVAILABLE = True
except ImportError:
    FACE_DETECTION_AVAILABLE = False
    logging.warning("Face detection not available")

try:
    from aspect_ratio_utils import AspectRatioProcessor
    ASPECT_RATIO_AVAILABLE = True
except ImportError:
    ASPECT_RATIO_AVAILABLE = False
    logging.warning("Aspect ratio processing not available")

try:
    from audio_processor import (
        extract_audio, sync_audio_to_video, add_background_music,
        normalize_audio, process_audio_for_video, check_video_has_audio
    )
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    logging.warning("Audio processing not available")

try:
    from export_utils import (
        export_custom_resolution, export_gif, export_frame_sequence,
        get_video_resolution
    )
    EXPORT_UTILS_AVAILABLE = True
except ImportError:
    EXPORT_UTILS_AVAILABLE = False
    logging.warning("Export utilities not available")

try:
    from retargeting_utils import (
        RetargetingParams, ExpressionPresets, apply_expression_intensity,
        calculate_blink_pattern, create_retargeting_params_from_preset
    )
    RETARGETING_UTILS_AVAILABLE = True
except ImportError:
    RETARGETING_UTILS_AVAILABLE = False
    logging.warning("Retargeting utilities not available")

try:
    from performance_utils import (
        GPUMemoryManager, ProcessingQueue, ProcessingCheckpoint,
        MultiGPUSupport, SystemMonitor, ProcessingState
    )
    PERFORMANCE_UTILS_AVAILABLE = True
except ImportError:
    PERFORMANCE_UTILS_AVAILABLE = False
    logging.warning("Performance utilities not available")

try:
    from history_manager import HistoryManager, GenerationHistory
    HISTORY_AVAILABLE = True
except ImportError:
    HISTORY_AVAILABLE = False
    logging.warning("History manager not available")

try:
    from ui_utils import (
        create_comparison_image, extract_video_frame,
        create_preview_grid, image_to_base64
    )
    UI_UTILS_AVAILABLE = True
except ImportError:
    UI_UTILS_AVAILABLE = False
    logging.warning("UI utilities not available")

# Paths - Use environment variable or default to current directory
project_root = Path(os.getenv('LIVEPORTRAIT_ROOT', Path.cwd()))
live_portrait_output_dir = project_root / 'output'
esrgan_input_dir = live_portrait_output_dir / 'frames'
esrgan_output_dir = live_portrait_output_dir / 'enhanced_frames'
esrgan_model_path = project_root / 'pretrained_weights' / 'RealESRGAN_x4plus_anime_6B.pth'
esrgan_script_path = project_root / 'real-esrgan' / 'inference_realesrgan.py'

# Ensure the output directory exists
os.makedirs(live_portrait_output_dir, exist_ok=True)
os.makedirs(esrgan_input_dir, exist_ok=True)
os.makedirs(esrgan_output_dir, exist_ok=True)

# Initialize enhancer (use enhanced version if available)
if ENHANCED_VIDEO_AVAILABLE:
    enhancer = EnhancedVideoEnhancer(esrgan_script_path, project_root / 'pretrained_weights', esrgan_output_dir)
else:
    enhancer = VideoEnhancer(esrgan_script_path, esrgan_model_path, esrgan_output_dir)

# Initialize performance utilities
if PERFORMANCE_UTILS_AVAILABLE:
    gpu_memory_manager = GPUMemoryManager(device_id=0, low_memory_mode=False)
    processing_queue = ProcessingQueue(max_size=100)
    checkpoint_manager = ProcessingCheckpoint(live_portrait_output_dir / 'checkpoints')
    multi_gpu = MultiGPUSupport()
    system_monitor = SystemMonitor()
    
    # Apply low-memory optimizations if needed
    if os.getenv('LOW_MEMORY_MODE', 'false').lower() == 'true':
        gpu_memory_manager.low_memory_mode = True
        gpu_memory_manager.optimize_for_low_memory()
else:
    gpu_memory_manager = None
    processing_queue = None
    checkpoint_manager = None
    multi_gpu = None
    system_monitor = None

# Initialize history manager
if HISTORY_AVAILABLE:
    history_manager = HistoryManager(live_portrait_output_dir / 'history', max_entries=100)
else:
    history_manager = None


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def downscale_video(input_video_path, output_video_path, width=1280):
    command = [
        'ffmpeg', '-i', str(input_video_path), '-vf', f'scale={width}:-1', str(output_video_path)
    ]
    subprocess.run(command, check=True)


def extract_frames(video_path, output_dir):
    command = [
        'ffmpeg', '-i', str(video_path),
        str(output_dir / 'frame_%04d.png')
    ]
    subprocess.run(command, check=True)


def reassemble_video(input_dir, output_video_path, fps=30):
    command = [
        'ffmpeg', '-framerate', str(fps), '-i',
        str(input_dir / 'frame_%04d.png'), '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p', str(output_video_path)
    ]
    subprocess.run(command, check=True)


def enhance_video(video_path, model_name='RealESRGAN_x4plus_anime_6B', 
                 quality='high', format='mp4', fps=30, smoothing_strength=0.0,
                 denoise_strength=0.0, stabilize=False, progress=gr.Progress()):
    """Enhanced video enhancement with model selection and export options."""
    try:
        if not video_path:
            raise ValueError("No video provided")
        
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        downscaled_video_path = live_portrait_output_dir / 'downscaled_video.mp4'
        
        # Generate output filename with format
        format_ext = format.lower() if format else 'mp4'
        enhanced_video_path = live_portrait_output_dir / f'enhanced_live_portrait.{format_ext}'
        
        # Get original video info for audio preservation
        audio_path = None
        if ENHANCED_VIDEO_AVAILABLE:
            video_info = get_video_info(video_path)
            original_fps = video_info.get('fps', 30)
            fps = fps if fps > 0 else original_fps
            
            # Extract audio if available
            if has_audio_stream(video_path):
                audio_path = live_portrait_output_dir / 'extracted_audio.wav'
                try:
                    ffmpeg_cmd = find_ffmpeg() if ENHANCED_VIDEO_AVAILABLE else 'ffmpeg'
                    cmd = [ffmpeg_cmd, '-i', str(video_path), '-vn', '-acodec', 'pcm_s16le', 
                          '-y', str(audio_path)]
                    subprocess.run(cmd, check=True, capture_output=True)
                except Exception as e:
                    logging.warning(f"Could not extract audio: {e}")
                    audio_path = None
        
        # Progress tracking
        def update_progress(progress_value, message):
            if progress:
                progress(progress_value, desc=message)

        # Downscale the video
        if ENHANCED_VIDEO_AVAILABLE:
            downscale_video(video_path, downscaled_video_path, width=1280, progress_callback=update_progress)
        else:
            downscale_video(video_path, downscaled_video_path)
            update_progress(0.3, "Video downscaled")

        # Extract frames
        if ENHANCED_VIDEO_AVAILABLE:
            extract_frames(downscaled_video_path, esrgan_input_dir, progress_callback=update_progress)
        else:
            extract_frames(downscaled_video_path, esrgan_input_dir)
            update_progress(0.5, "Frames extracted")

        # Enhance frames using Real-ESRGAN
        if ENHANCED_VIDEO_AVAILABLE and hasattr(enhancer, 'enhance_frames'):
            enhancer.enhance_frames(esrgan_input_dir, model_name=model_name, progress_callback=update_progress)
        else:
            enhancer.enhance_frames(esrgan_input_dir)
            update_progress(0.8, "Frames enhanced")

        # Reassemble the enhanced frames into a video
        if ENHANCED_VIDEO_AVAILABLE:
            reassemble_video(esrgan_output_dir, enhanced_video_path, fps=fps, 
                           quality=quality, format=format_ext, audio_path=audio_path,
                           smoothing_strength=smoothing_strength, 
                           denoise_strength=denoise_strength, 
                           stabilize=stabilize,
                           progress_callback=update_progress)
        else:
            reassemble_video(esrgan_output_dir, enhanced_video_path, fps=fps)
            update_progress(1.0, "Video complete!")
        
        return str(enhanced_video_path)
    
    except Exception as e:
        logging.error(f"Error enhancing video: {str(e)}")
        raise gr.Error(f"Failed to enhance video: {str(e)}")


# set tyro theme
tyro.extras.set_accent_color("bright_cyan")
args = tyro.cli(ArgumentConfig)

# specify configs for inference
inference_cfg = partial_fields(InferenceConfig, args.__dict__)  # use attribute of args to initial InferenceConfig
crop_cfg = partial_fields(CropConfig, args.__dict__)  # use attribute of args to initial CropConfig

gradio_pipeline = GradioPipeline(
    inference_cfg=inference_cfg,
    crop_cfg=crop_cfg,
    args=args
)


def detect_faces_interface(image_path, progress=gr.Progress()):
    """Interface for face detection."""
    if not image_path:
        return None, gr.update(visible=False), gr.update(choices=[], visible=False)
    
    try:
        if progress:
            progress(0.1, "Detecting faces...")
        
        # Initialize detector with pretrained weights path
        model_path = project_root / 'pretrained_weights' / 'insightface' / 'models' / 'buffalo_l'
        
        detector = FaceDetector(str(model_path) if model_path.exists() else None)
        faces = detector.detect_faces(image_path)
        
        if progress:
            progress(1.0, f"Detected {len(faces)} face(s)")
        
        if not faces:
            return None, gr.update(visible=False), gr.update(
                choices=[], 
                visible=False,
                value=[]
            )
        
        # Draw face boxes
        annotated_image = detector.draw_face_boxes(image_path, faces)
        
        # Save annotated image temporarily
        if CV2_AVAILABLE:
            temp_path = live_portrait_output_dir / 'face_detection_preview.jpg'
            cv2.imwrite(str(temp_path), annotated_image)
        else:
            temp_path = image_path  # Fallback
        
        # Create face selection choices
        face_choices = [f"Face {i+1} (Confidence: {f['confidence']:.2f})" for i, f in enumerate(faces)]
        
        return (
            str(temp_path),
            gr.update(visible=True),
            gr.update(choices=face_choices, visible=True, value=face_choices if len(faces) == 1 else [])
        )
    
    except Exception as e:
        logging.error(f"Face detection error: {str(e)}")
        return None, gr.update(visible=False), gr.update(choices=[], visible=False)


def gpu_wrapped_execute_video(image_path, video_path, relative_motion, do_crop, remap,
                              crop_driving_video, smoothing_strength=0.0, denoise_strength=0.0,
                              stabilize=False, selected_face_indices=None,
                              aspect_ratio='1:1', custom_width=1024, custom_height=1024,
                              crop_mode='center', preserve_bg=True,
                              enable_audio_sync=True, add_background_music=False,
                              background_music_file=None, music_volume=0.3,
                              original_audio_volume=1.0, normalize_audio=True,
                              loop_audio=False, progress=gr.Progress(),
                              enable_checkpoint=False, job_id=None):
    """Wrapper for video execution with smoothing, multi-face, aspect ratio, audio, and performance options."""
    
    # Performance optimizations
    if PERFORMANCE_UTILS_AVAILABLE:
        # Clear cache before processing
        if gpu_memory_manager:
            gpu_memory_manager.clear_cache()
            if progress:
                mem_info = gpu_memory_manager.get_memory_info_string()
                progress(0.01, f"GPU Memory: {mem_info}")
        
        # Create job ID if not provided
        if job_id is None:
            import uuid
            job_id = str(uuid.uuid4())[:8]
        
        # Create checkpoint state
        if enable_checkpoint and checkpoint_manager:
            state = ProcessingState(
                job_id=job_id,
                image_path=str(image_path),
                video_path=str(video_path),
                parameters={
                    'relative_motion': relative_motion,
                    'do_crop': do_crop,
                    'remap': remap,
                    'crop_driving_video': crop_driving_video,
                },
                current_step='initialization',
                progress=0.0
            )
            checkpoint_manager.save_checkpoint(state)
    # Handle aspect ratio preprocessing
    processed_image_path = image_path
    if ASPECT_RATIO_AVAILABLE and aspect_ratio != '1:1':
        try:
            processor = AspectRatioProcessor()
            
            # Get face bbox if available for face-aware cropping
            face_bbox = None
            if crop_mode == 'face' and FACE_DETECTION_AVAILABLE:
                try:
                    model_path = project_root / 'pretrained_weights' / 'insightface' / 'models' / 'buffalo_l'
                    detector = FaceDetector(str(model_path) if model_path.exists() else None)
                    faces = detector.detect_faces(image_path)
                    if faces:
                        face_bbox = faces[0]['bbox']  # Use first face
                except:
                    pass
            
            # Process image to target aspect ratio
            processed_img = processor.resize_to_aspect(
                image_path,
                aspect_ratio,
                int(custom_width) if custom_width else None,
                int(custom_height) if custom_height else None,
                max_dimension=1024,
                preserve_background=preserve_bg,
                crop_mode=crop_mode,
                face_bbox=face_bbox
            )
            
            # Save processed image
            processed_image_path = live_portrait_output_dir / 'aspect_processed_image.jpg'
            if CV2_AVAILABLE:
                cv2.imwrite(str(processed_image_path), processed_img)
            else:
                processed_image_path = image_path  # Fallback
        except Exception as e:
            logging.warning(f"Aspect ratio processing failed: {e}, using original image")
            processed_image_path = image_path
    
    # Handle multi-face processing if faces are selected
    if FACE_DETECTION_AVAILABLE and selected_face_indices and len(selected_face_indices) > 0:
        try:
            # Process each selected face
            results = []
            model_path = project_root / 'pretrained_weights' / 'insightface' / 'models' / 'buffalo_l'
            detector = FaceDetector(str(model_path) if model_path.exists() else None)
            faces = detector.detect_faces(image_path)
            
            selected_faces_list = [faces[int(idx.split()[1]) - 1] for idx in selected_face_indices 
                                 if idx.startswith("Face")]
            
            for idx, face in enumerate(selected_faces_list):
                if progress:
                    progress(idx / len(selected_faces_list), f"Processing face {idx+1}/{len(selected_faces_list)}...")
                
                # Crop face from image
                if CV2_AVAILABLE:
                    cropped_face = detector.crop_face(image_path, face)
                    temp_face_path = live_portrait_output_dir / f'temp_face_{idx}.jpg'
                    cv2.imwrite(str(temp_face_path), cropped_face)
                    
                    # Process this face (use processed image if aspect ratio was applied)
                    face_img_path = str(temp_face_path)
                    result = gradio_pipeline.execute_video(
                        face_img_path, video_path, relative_motion, 
                        False, remap, crop_driving_video  # Don't crop already cropped face
                    )
                    
                    if isinstance(result, tuple):
                        results.append(result[1] if len(result) > 1 else result[0])
                    else:
                        results.append(result)
            
            # For now, return the first result (TODO: composite multiple faces)
            if results:
                result = results[0]
            else:
                result = gradio_pipeline.execute_video(image_path, video_path, relative_motion, 
                                                       do_crop, remap, crop_driving_video)
        except Exception as e:
            logging.warning(f"Multi-face processing failed: {e}, falling back to single face")
            result = gradio_pipeline.execute_video(processed_image_path, video_path, relative_motion, 
                                                   do_crop, remap, crop_driving_video)
    else:
        # Standard single-face processing (use processed image if aspect ratio was applied)
        result = gradio_pipeline.execute_video(processed_image_path, video_path, relative_motion, 
                                               do_crop, remap, crop_driving_video)
    
    # Handle tuple return (original, cropped)
    if isinstance(result, tuple):
        video_path_output = result[1] if len(result) > 1 else result[0]
    else:
        video_path_output = result
    
    # Apply smoothing if requested
    if (smoothing_strength > 0 or denoise_strength > 0 or stabilize) and ENHANCED_VIDEO_AVAILABLE:
        try:
            from video_enhanced import apply_smoothing_filters
            if progress:
                progress(0.95, "Applying smoothing filters...")
            apply_smoothing_filters(video_path_output, smoothing_strength, 
                                  denoise_strength, stabilize, 
                                  lambda p, m: progress(p, desc=m) if progress else None)
        except Exception as e:
            logging.warning(f"Smoothing failed: {str(e)}")
    
    # Handle audio processing
    if AUDIO_PROCESSING_AVAILABLE:
        try:
            # Determine which video to use for audio processing
            if isinstance(result, tuple):
                video_to_process = result[1] if len(result) > 1 else result[0]
            else:
                video_to_process = result
            
            video_to_process = str(video_to_process)
            
            # Process audio if enabled
            if enable_audio_sync or (add_background_music and background_music_file):
                if progress:
                    progress(0.98, "Processing audio...")
                
                # Get background music path
                music_path = None
                if add_background_music and background_music_file:
                    music_path = background_music_file.name if hasattr(background_music_file, 'name') else str(background_music_file)
                
                # Process audio
                audio_output = process_audio_for_video(
                    video_to_process,
                    source_video_path=str(video_path) if enable_audio_sync else None,
                    background_music_path=music_path,
                    music_volume=float(music_volume),
                    normalize=normalize_audio,
                    loop_audio=loop_audio
                )
                
                if audio_output and os.path.exists(audio_output):
                    # Replace result with audio-enhanced version
                    if isinstance(result, tuple):
                        result = (result[0], audio_output) if len(result) > 1 else (audio_output,)
                    else:
                        result = audio_output
                    
                    if progress:
                        progress(1.0, "Audio processing complete!")
        except Exception as e:
            logging.warning(f"Audio processing failed: {str(e)}")
    
    return result


def gpu_wrapped_execute_image(*args, **kwargs):
    return gradio_pipeline.execute_image(*args, **kwargs)


# assets
example_portrait_dir = "assets/examples/source"
example_video_dir = "assets/examples/driving"
data_examples = [
    [osp.join(example_portrait_dir, "s9.jpg"), osp.join(example_video_dir, "d0.mp4"), True, True, True, False],
    [osp.join(example_portrait_dir, "s6.jpg"), osp.join(example_video_dir, "d0.mp4"), True, True, True, False],
    [osp.join(example_portrait_dir, "s10.jpg"), osp.join(example_video_dir, "d0.mp4"), True, True, True, False],
    [osp.join(example_portrait_dir, "s5.jpg"), osp.join(example_video_dir, "d18.mp4"), True, True, True, False],
    [osp.join(example_portrait_dir, "s7.jpg"), osp.join(example_video_dir, "d19.mp4"), True, True, True, False],
    [osp.join(example_portrait_dir, "s2.jpg"), osp.join(example_video_dir, "d13.mp4"), True, True, True, True],
]

#################### interface logic ####################

def enhance_video_interface(video_path, model_display_name=None, 
                           quality='high', format='mp4', fps=30,
                           smoothing_strength=0.0, denoise_strength=0.0, stabilize=False,
                           export_custom_resolution=False, export_width=1920, export_height=1080,
                           export_maintain_aspect=True, export_gif=False, gif_fps=15, gif_width=512,
                           export_frames=False, frame_format='png', frame_step=1,
                           progress=gr.Progress()):
    """Interface wrapper for video enhancement with export options."""
    try:
        # Map display name to model ID
        if ENHANCED_VIDEO_AVAILABLE and hasattr(enhancer, 'get_available_models'):
            available_models = enhancer.get_available_models()
            model_map = {m['description']: m['id'] for m in available_models}
            model_name = model_map.get(model_display_name, 'RealESRGAN_x4plus_anime_6B')
        else:
            model_name = 'RealESRGAN_x4plus_anime_6B'
        
        enhanced_video_path = enhance_video(video_path, model_name, quality, format, fps,
                                          smoothing_strength, denoise_strength, stabilize, progress)
        
        # Handle export options
        export_results = []
        
        if EXPORT_UTILS_AVAILABLE:
            base_path = Path(enhanced_video_path)
            output_dir = base_path.parent
            
            # Custom resolution export
            if export_custom_resolution:
                if progress:
                    progress(0.95, "Exporting custom resolution...")
                custom_output = output_dir / f"{base_path.stem}_custom_{int(export_width)}x{int(export_height)}.mp4"
                if export_custom_resolution(enhanced_video_path, str(custom_output), 
                                          int(export_width), int(export_height),
                                          export_maintain_aspect, quality):
                    export_results.append(str(custom_output))
            
            # GIF export
            if export_gif:
                if progress:
                    progress(0.96, "Exporting GIF...")
                gif_output = output_dir / f"{base_path.stem}.gif"
                if export_gif(enhanced_video_path, str(gif_output), int(gif_fps), int(gif_width)):
                    export_results.append(str(gif_output))
            
            # Frame sequence export
            if export_frames:
                if progress:
                    progress(0.97, "Exporting frame sequence...")
                frames_dir = output_dir / f"{base_path.stem}_frames"
                success, frame_count = export_frame_sequence(
                    enhanced_video_path, str(frames_dir), frame_format, 
                    'frame', 0, None, int(frame_step)
                )
                if success:
                    export_results.append(str(frames_dir))
        
        # Return main video (export results are saved to disk)
        return enhanced_video_path
    except Exception as e:
        logging.error(f"Enhancement error: {str(e)}")
        raise gr.Error(f"Enhancement failed: {str(e)}")


def process_batch(images, videos, relative_motion, do_crop, remap, 
                  enhance_enabled, enhance_model, enhance_quality, progress=gr.Progress()):
    """Process multiple images and videos in batch with queue system and per-item tracking."""
    try:
        from batch_processor import BatchProcessor
        BATCH_PROCESSOR_AVAILABLE = True
    except ImportError:
        BATCH_PROCESSOR_AVAILABLE = False
    
    import zipfile
    from datetime import datetime
    import time
    
    if not images or not videos:
        return "Error: Please upload at least one image and one video.", None, gr.update(visible=False), gr.update(visible=False)
    
    batch_output_dir = live_portrait_output_dir / 'batch_output'
    batch_output_dir.mkdir(exist_ok=True)
    
    # Initialize batch processor if available
    if BATCH_PROCESSOR_AVAILABLE:
        processor = BatchProcessor(batch_output_dir)
        items = processor.create_batch_queue(images, videos)
        total_items = len(items)
    else:
        total_items = len(images) * len(videos)
        items = None
    
    results = []
    status_updates = []
    
    # Map model display name to model ID
    model_name = 'RealESRGAN_x4plus_anime_6B'
    if enhance_enabled and enhance_model:
        if ENHANCED_VIDEO_AVAILABLE and hasattr(enhancer, 'get_available_models'):
            try:
                available_models = enhancer.get_available_models()
                for model in available_models:
                    if model['name'] in enhance_model or model['description'] in enhance_model:
                        model_name = model['id']
                        break
            except:
                pass
    
    try:
        current_item = 0
        
        for img_idx, image_file in enumerate(images):
            for vid_idx, video_file in enumerate(videos):
                current_item += 1
                progress_value = current_item / total_items
                
                # Update item status if using batch processor
                if BATCH_PROCESSOR_AVAILABLE and items:
                    item = items[current_item - 1]
                    item.status = 'processing'
                    item.start_time = time.time()
                
                # Detailed status message
                status_msg = f"[{current_item}/{total_items}] Processing Image {img_idx+1} × Video {vid_idx+1}..."
                status_updates.append(status_msg)
                
                if progress:
                    progress(progress_value, desc=status_msg)
                
                # Get status text if using batch processor
                if BATCH_PROCESSOR_AVAILABLE and items:
                    status_text = processor.get_item_status_text()
                    if progress:
                        progress(progress_value, desc=status_text.split('\n')[0] if status_text else status_msg)
                else:
                    status_text = "\n".join(status_updates[-5:])  # Show last 5 updates
                
                try:
                    # Get file paths
                    image_path = image_file.name if hasattr(image_file, 'name') else str(image_file)
                    video_path_input = video_file.name if hasattr(video_file, 'name') else str(video_file)
                    
                    # Process animation
                    output_video = gradio_pipeline.execute_video(
                        image_path,
                        video_path_input,
                        relative_motion,
                        do_crop,
                        remap,
                        False  # crop_driving_video
                    )
                    
                    # Handle tuple return (original, cropped)
                    if isinstance(output_video, tuple):
                        output_video = output_video[1] if len(output_video) > 1 else output_video[0]
                    
                    if output_video and os.path.exists(str(output_video)):
                        final_path = str(output_video)
                        
                        # Enhance if enabled
                        if enhance_enabled:
                            try:
                                enhanced_path = enhance_video(
                                    final_path,
                                    model_name,
                                    enhance_quality,
                                    'mp4',
                                    30,
                                    progress
                                )
                                if enhanced_path and os.path.exists(str(enhanced_path)):
                                    results.append(str(enhanced_path))
                                    if BATCH_PROCESSOR_AVAILABLE and items:
                                        item.result_path = str(enhanced_path)
                                        item.status = 'completed'
                                        item.end_time = time.time()
                                    status_updates.append(f"✅ [{current_item}/{total_items}] Enhanced successfully")
                                else:
                                    results.append(final_path)
                                    if BATCH_PROCESSOR_AVAILABLE and items:
                                        item.result_path = final_path
                                        item.status = 'completed'
                                        item.end_time = time.time()
                                    status_updates.append(f"✅ [{current_item}/{total_items}] Completed (enhancement skipped)")
                            except Exception as e:
                                results.append(final_path)
                                if BATCH_PROCESSOR_AVAILABLE and items:
                                    item.result_path = final_path
                                    item.status = 'completed'
                                    item.end_time = time.time()
                                    item.error = f"Enhancement failed: {str(e)}"
                                status_updates.append(f"⚠️ [{current_item}/{total_items}] Completed (enhancement failed)")
                        else:
                            results.append(final_path)
                            if BATCH_PROCESSOR_AVAILABLE and items:
                                item.result_path = final_path
                                item.status = 'completed'
                                item.end_time = time.time()
                            status_updates.append(f"✅ [{current_item}/{total_items}] Completed")
                    else:
                        if BATCH_PROCESSOR_AVAILABLE and items:
                            item.status = 'failed'
                            item.error = "No output generated"
                            item.end_time = time.time()
                        status_updates.append(f"❌ [{current_item}/{total_items}] Failed: No output")
                
                except Exception as e:
                    error_msg = f"❌ [{current_item}/{total_items}] Error: {str(e)}"
                    if BATCH_PROCESSOR_AVAILABLE and items:
                        item.status = 'failed'
                        item.error = str(e)
                        item.end_time = time.time()
                    status_updates.append(error_msg)
                    logging.error(error_msg, exc_info=True)
        
        # Update processor results
        if BATCH_PROCESSOR_AVAILABLE and items:
            processor.results = results
            status_text = processor.get_item_status_text()
        else:
            status_text = "\n".join(status_updates)
        
        # Create ZIP file if we have results
        zip_path = None
        if results:
            if BATCH_PROCESSOR_AVAILABLE and items:
                zip_path = processor.create_zip_archive()
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                zip_path = batch_output_dir / f"batch_results_{timestamp}.zip"
                
                try:
                    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for idx, result_file in enumerate(results):
                            if os.path.exists(result_file):
                                arcname = f"result_{idx+1:03d}_{os.path.basename(result_file)}"
                                zipf.write(result_file, arcname)
                    zip_path = str(zip_path)
                except Exception as e:
                    logging.error(f"ZIP creation failed: {str(e)}")
                    zip_path = None
            
            if zip_path:
                status_text += f"\n\n✅ Batch processing complete! {len(results)}/{total_items} files processed successfully."
                status_text += f"\n📦 ZIP archive created: {os.path.basename(zip_path)}"
            else:
                status_text += f"\n\n✅ Batch processing complete! {len(results)}/{total_items} files processed."
                status_text += f"\n⚠️ ZIP creation failed, but individual files are available."
        else:
            status_text += "\n\n❌ Batch processing completed with errors. No files generated."
        
        return (
            status_text,
            results if results else None,
            gr.update(visible=bool(results)),
            gr.update(visible=bool(zip_path))
        )
    
    except Exception as e:
        error_msg = f"Batch processing failed: {str(e)}"
        logging.error(error_msg, exc_info=True)
        return error_msg, None, gr.update(visible=False), gr.update(visible=False)


# Modern CSS styling - Dark Theme matching Chat-with-Ollama/MoA
custom_css = """
    :root {
        --dark-bg: #0d1117;
        --dark-surface: #161b22;
        --dark-surface-hover: #1c2128;
        --glass-bg: rgba(255, 255, 255, 0.03);
        --glass-border: rgba(255, 255, 255, 0.08);
        --text-primary: #e6edf3;
        --text-secondary: rgba(230, 237, 243, 0.6);
        --accent: #58a6ff;
        --accent-hover: #79c0ff;
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }
    
    .gradio-container {
        max-width: 1400px !important;
        margin: auto;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        background: var(--dark-bg) !important;
        color: var(--text-primary) !important;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: var(--text-secondary);
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .section-header {
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--glass-border);
    }
    
    .info-text {
        background: var(--glass-bg);
        backdrop-filter: blur(20px);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid var(--glass-border);
        border-left: 4px solid var(--accent);
        margin: 1rem 0;
        color: var(--text-secondary);
        font-size: 0.95rem;
    }
    
    .button-group {
        display: flex;
        gap: 0.75rem;
        flex-wrap: wrap;
        justify-content: center;
        margin: 1.5rem 0;
    }
    
    .output-section {
        margin-top: 2rem;
        padding-top: 2rem;
        border-top: 2px solid var(--glass-border);
    }
    
    /* Dark theme overrides for Gradio components */
    .gradio-container .gr-form,
    .gradio-container .gr-box,
    .gradio-container .gr-panel {
        background: var(--dark-surface) !important;
        border-color: var(--glass-border) !important;
        color: var(--text-primary) !important;
    }
    
    .gradio-container input,
    .gradio-container textarea,
    .gradio-container select {
        background: var(--glass-bg) !important;
        border-color: var(--glass-border) !important;
        color: var(--text-primary) !important;
    }
    
    .gradio-container .gr-button-primary {
        background: var(--primary-gradient) !important;
        border: none !important;
    }
    
    .gradio-container .gr-button-primary:hover {
        opacity: 0.9;
        transform: translateY(-2px);
    }
    
    .monospace {
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        line-height: 1.4;
    }
"""

# Create modern dark theme matching Chat-with-Ollama/MoA
theme = gr.themes.Soft(
    primary_hue="purple",
    secondary_hue="slate",
    font=("Inter", "ui-sans-serif", "system-ui", "sans-serif"),
).set(
    body_background_fill="#0d1117",
    body_background_fill_dark="#0d1117",
    block_background_fill="#161b22",
    block_background_fill_dark="#161b22",
    block_border_color="rgba(255, 255, 255, 0.08)",
    block_border_color_dark="rgba(255, 255, 255, 0.08)",
    button_primary_background_fill="#667eea",
    button_primary_background_fill_hover="#5568d3",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="rgba(255, 255, 255, 0.03)",
    button_secondary_background_fill_hover="rgba(255, 255, 255, 0.05)",
    button_secondary_text_color="#e6edf3",
    border_color_accent="#667eea",
    shadow_drop_lg="0 8px 24px rgba(0, 0, 0, 0.5)",
    input_background_fill="rgba(255, 255, 255, 0.03)",
    input_background_fill_dark="rgba(255, 255, 255, 0.03)",
    input_border_color="rgba(255, 255, 255, 0.08)",
    input_border_color_dark="rgba(255, 255, 255, 0.08)",
    input_text_color="#e6edf3",
    input_text_color_dark="#e6edf3",
    body_text_color="#e6edf3",
    body_text_color_dark="#e6edf3",
    block_label_text_color="#e6edf3",
    block_label_text_color_dark="#e6edf3",
)

with gr.Blocks(theme=theme, css=custom_css, title="PresentaPulse - LivePortrait Animation") as demo:
    # Header
    gr.HTML("""
        <div class="main-header">
            <h1 style="margin: 0; font-size: 2.5rem;">🎬 PresentaPulse</h1>
        </div>
    """)
    gr.Markdown(
        '<p class="sub-header">Efficient Portrait Animation with Smoothing and Retargeting Control</p>',
        elem_classes=["sub-header"]
    )
    
    # Performance Monitoring (if available)
    if PERFORMANCE_UTILS_AVAILABLE:
        with gr.Accordion("⚡ Performance & System Info", open=False, elem_id="performance_panel"):
            with gr.Row():
                with gr.Column():
                    performance_status = gr.Markdown(
                        value="**System Status:**\n- GPU Memory: Loading...\n- CPU Usage: Loading...\n- RAM Usage: Loading...",
                        label="System Status"
                    )
                    
                    refresh_performance = gr.Button("🔄 Refresh Status", size="sm")
                    
                    low_memory_mode_toggle = gr.Checkbox(
                        value=False,
                        label="Low Memory Mode",
                        info="Enable for GPUs with limited VRAM (<6GB)"
                    )
                    
                    if multi_gpu and len(multi_gpu.get_available_devices()) > 1:
                        gpu_selection = gr.Radio(
                            choices=[f"GPU {i}" for i in multi_gpu.get_available_devices()],
                            value=f"GPU {multi_gpu.get_available_devices()[0]}",
                            label="Select GPU",
                            info="Choose which GPU to use for processing"
                        )
                    else:
                        gpu_selection = gr.Radio(visible=False)
                    
                    def update_performance_status():
                        if not PERFORMANCE_UTILS_AVAILABLE:
                            return "Performance monitoring not available"
                        
                        info = system_monitor.get_system_info()
                        status_lines = ["**System Status:**"]
                        
                        # CPU and RAM
                        status_lines.append(f"- **CPU Usage:** {info['cpu_usage']:.1f}%")
                        ram = info['ram']
                        status_lines.append(f"- **RAM:** {ram['used']:.0f}MB / {ram['total']:.0f}MB ({ram['percent']:.1f}%)")
                        
                        # GPU Info
                        if 'gpus' in info:
                            status_lines.append(f"\n**GPU Information:**")
                            for gpu in info['gpus']:
                                mem = gpu['memory']
                                status_lines.append(
                                    f"- **{gpu['name']} (GPU {gpu['id']}):** "
                                    f"{mem['allocated']:.0f}MB / {mem['total']:.0f}MB "
                                    f"({mem['allocated']/mem['total']*100:.1f}% used)"
                                )
                        
                        # Queue Status
                        if processing_queue:
                            queue_size = processing_queue.size()
                            status_lines.append(f"\n**Processing Queue:** {queue_size} job(s) pending")
                        
                        return "\n".join(status_lines)
                    
                    def toggle_low_memory_mode(enabled):
                        if PERFORMANCE_UTILS_AVAILABLE and gpu_memory_manager:
                            gpu_memory_manager.low_memory_mode = enabled
                            if enabled:
                                gpu_memory_manager.optimize_for_low_memory()
                            else:
                                gpu_memory_manager.clear_cache()
                        return update_performance_status()
                    
                    refresh_performance.click(
                        fn=update_performance_status,
                        outputs=[performance_status]
                    )
                    
                    low_memory_mode_toggle.change(
                        fn=toggle_low_memory_mode,
                        inputs=[low_memory_mode_toggle],
                        outputs=[performance_status]
                    )
                    
                    # Initial status update
                    demo.load(
                        fn=update_performance_status,
                        outputs=[performance_status]
                    )
    
    # Main interface with Tabs
    with gr.Tabs() as main_tabs:
        # Animation Tab
        with gr.Tab("🎥 Animation", id="animation_tab"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.Markdown("### 📸 Source Portrait", elem_classes=["section-header"])
                    image_input = gr.Image(
                        type="filepath",
                        label="Upload or select a portrait image",
                        height=400,
                        show_label=True
                    )
                    
                    # Face detection controls
                    if FACE_DETECTION_AVAILABLE:
                        with gr.Row():
                            detect_faces_button = gr.Button(
                                "🔍 Detect Faces",
                                variant="secondary",
                                size="sm"
                            )
                            face_selection_mode = gr.Checkbox(
                                value=False,
                                label="Multi-Face Mode",
                                info="Enable to select multiple faces"
                            )
                        
                        face_detection_result = gr.Image(
                            label="Face Detection Preview",
                            height=300,
                            visible=False,
                            show_label=True
                        )
                        
                        selected_faces = gr.CheckboxGroup(
                            choices=[],
                            label="Select Faces to Animate",
                            visible=False,
                            info="Select which faces to animate (leave empty for all)"
                        )
                    else:
                        detect_faces_button = gr.Button(visible=False)
                        face_selection_mode = gr.Checkbox(visible=False)
                        face_detection_result = gr.Image(visible=False)
                        selected_faces = gr.CheckboxGroup(visible=False)
                    gr.Examples(
                        examples=[
                            [osp.join(example_portrait_dir, "s9.jpg")],
                            [osp.join(example_portrait_dir, "s6.jpg")],
                            [osp.join(example_portrait_dir, "s10.jpg")],
                            [osp.join(example_portrait_dir, "s5.jpg")],
                            [osp.join(example_portrait_dir, "s7.jpg")],
                            [osp.join(example_portrait_dir, "s12.jpg")],
                        ],
                        inputs=[image_input],
                        cache_examples=False,
                        label="Example Portraits"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🎬 Driving Video", elem_classes=["section-header"])
                    video_input = gr.Video(
                        label="Upload or select a driving video",
                        height=400,
                        show_label=True
                    )
                    gr.Examples(
                        examples=[
                            [osp.join(example_video_dir, "d0.mp4")],
                            [osp.join(example_video_dir, "d18.mp4")],
                            [osp.join(example_video_dir, "d19.mp4")],
                            [osp.join(example_video_dir, "d14.mp4")],
                            [osp.join(example_video_dir, "d6.mp4")],
                        ],
                        inputs=[video_input],
                        cache_examples=False,
                        label="Example Videos"
                    )
            
            with gr.Accordion("⚙️ Animation Settings", open=False):
                gr.Markdown(
                    '<p class="info-text">Configure animation options to control the behavior of the portrait animation.</p>',
                    elem_classes=["info-text"]
                )
                with gr.Row():
                    flag_relative_input = gr.Checkbox(
                        value=True,
                        label="Relative Motion",
                        info="Use relative motion for more natural animations"
                    )
                    flag_do_crop_input = gr.Checkbox(
                        value=True,
                        label="Crop Source Image",
                        info="Crop the source portrait before processing"
                    )
                with gr.Row():
                    flag_remap_input = gr.Checkbox(
                        value=True,
                        label="Paste Back",
                        info="Paste the animated result back into the original image"
                    )
                    flag_crop_driving_video_input = gr.Checkbox(
                        value=False,
                        label="Crop Driving Video",
                        info="Crop the driving video before processing"
                    )
                
                gr.Markdown("---")
                gr.Markdown("### 🎬 Advanced Animation Controls", elem_classes=["section-header"])
                
                with gr.Row():
                    animation_speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        label="Animation Speed",
                        info="1.0 = normal speed, <1.0 = slow motion, >1.0 = fast forward"
                    )
                    animation_fps = gr.Slider(
                        minimum=15,
                        maximum=60,
                        step=1,
                        value=30,
                        label="Output Frame Rate (FPS)",
                        info="Higher FPS = smoother animation but larger file size"
                    )
                
                with gr.Row():
                    loop_animation = gr.Checkbox(
                        value=False,
                        label="Loop Animation",
                        info="Create seamless looping animation"
                    )
                    preview_mode = gr.Checkbox(
                        value=False,
                        label="Quick Preview Mode",
                        info="Generate lower quality preview first (faster)"
                    )
                
                gr.Markdown("---")
                gr.Markdown("### 🎨 Smoothing & Denoising", elem_classes=["section-header"])
                
                with gr.Row():
                    smoothing_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.0,
                        label="Temporal Smoothing",
                        info="Reduces jitter between frames (0 = off, 1.0 = maximum smoothing)"
                    )
                    denoise_strength = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.1,
                        value=0.0,
                        label="Denoising Strength",
                        info="Removes noise from frames (0 = off, 1.0 = maximum denoising)"
                    )
                
                with gr.Row():
                    stabilize_motion = gr.Checkbox(
                        value=False,
                        label="Motion Stabilization",
                        info="Reduces camera shake and unwanted motion"
                    )
                    
                    gr.Markdown(
                        '<p class="info-text" style="font-size: 0.85rem; margin-top: 8px;">💡 <strong>Tip:</strong> Use smoothing for jittery animations, denoising for noisy videos, and stabilization for shaky footage. Higher values = more processing time.</p>',
                        elem_classes=["info-text"]
                    )
                
                gr.Markdown("---")
                gr.Markdown("### 📐 Aspect Ratio & Cropping", elem_classes=["section-header"])
                
                if ASPECT_RATIO_AVAILABLE:
                    aspect_ratio = gr.Radio(
                        choices=['1:1', '16:9', '9:16', '4:3', '21:9', 'custom'],
                        value='1:1',
                        label="Output Aspect Ratio",
                        info="Choose output format (1:1 = Square, 16:9 = Widescreen, 9:16 = Portrait)"
                    )
                    
                    with gr.Row(visible=False) as custom_aspect_row:
                        custom_width = gr.Number(
                            value=1024,
                            label="Custom Width",
                            minimum=256,
                            maximum=4096,
                            step=64
                        )
                        custom_height = gr.Number(
                            value=1024,
                            label="Custom Height",
                            minimum=256,
                            maximum=4096,
                            step=64
                        )
                    
                    aspect_ratio.change(
                        fn=lambda x: gr.update(visible=(x == 'custom')),
                        inputs=[aspect_ratio],
                        outputs=[custom_aspect_row]
                    )
                    
                    crop_mode = gr.Radio(
                        choices=['center', 'face', 'top', 'bottom', 'left', 'right'],
                        value='center',
                        label="Crop Mode",
                        info="How to crop the image (face = face-aware, center = center crop)"
                    )
                    
                    preserve_bg = gr.Checkbox(
                        value=True,
                        label="Preserve Background",
                        info="Add padding instead of cropping (maintains full image)"
                    )
                else:
                    aspect_ratio = gr.Radio(visible=False)
                    custom_width = gr.Number(visible=False)
                    custom_height = gr.Number(visible=False)
                    crop_mode = gr.Radio(visible=False)
                    preserve_bg = gr.Checkbox(visible=False)
            
            with gr.Row(elem_classes=["button-group"]):
                process_button_animation = gr.Button(
                    "🚀 Generate Animation",
                    variant="primary",
                    size="lg",
                    scale=2
                )
                process_button_reset = gr.ClearButton(
                    [image_input, video_input],
                    value="🧹 Clear All",
                    size="lg"
                )
            
            gr.Markdown("---")
            gr.Markdown("### 📹 Output Videos", elem_classes=["section-header"])
            
            with gr.Row():
                with gr.Column():
                    output_video = gr.Video(
                        label="Animated Video (Original Space)",
                        show_label=True,
                        height=400
                    )
                with gr.Column():
                    output_video_concat = gr.Video(
                        label="Animated Video (Cropped)",
                        show_label=True,
                        height=400
                    )
            
            # Real-time Preview (if available)
            if UI_UTILS_AVAILABLE:
                gr.Markdown("---")
                gr.Markdown("### 👁️ Real-time Preview", elem_classes=["section-header"])
                
                preview_enabled = gr.Checkbox(
                    value=False,
                    label="Enable Real-time Preview",
                    info="Show preview frames during processing"
                )
                
                preview_frame = gr.Image(
                    label="Preview Frame",
                    height=300,
                    visible=False,
                    show_label=True
                )
                
                preview_enabled.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[preview_enabled],
                    outputs=[preview_frame]
                )
            
            # Comparison View (if available)
            if UI_UTILS_AVAILABLE:
                gr.Markdown("---")
                gr.Markdown("### 🔍 Comparison View", elem_classes=["section-header"])
                
                show_comparison = gr.Checkbox(
                    value=False,
                    label="Show Before/After Comparison",
                    info="Compare original image with animated result"
                )
                
                comparison_image = gr.Image(
                    label="Before / After Comparison",
                    height=400,
                    visible=False,
                    show_label=True
                )
                
                show_comparison.change(
                    fn=lambda x: gr.update(visible=x),
                    inputs=[show_comparison],
                    outputs=[comparison_image]
                )
            
            # Audio Integration Section
            if AUDIO_PROCESSING_AVAILABLE:
                gr.Markdown("---")
                gr.Markdown("### 🔊 Audio Integration", elem_classes=["section-header"])
                
                with gr.Accordion("🎵 Audio Settings", open=False):
                    enable_audio_sync = gr.Checkbox(
                        value=True,
                        label="Sync Audio from Driving Video",
                        info="Extract and sync audio from the driving video to the animation"
                    )
                    
                    add_background_music_enabled = gr.Checkbox(
                        value=False,
                        label="Add Background Music",
                        info="Add background music to the final video"
                    )
                    
                    background_music_file = gr.File(
                        file_types=["audio"],
                        label="Background Music File",
                        visible=False,
                        info="Upload an audio file (MP3, WAV, etc.)"
                    )
                    
                    with gr.Row(visible=False) as music_settings_row:
                        music_volume = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.3,
                            label="Music Volume",
                            info="Volume of background music (0.0 = silent, 1.0 = full volume)"
                        )
                        original_audio_volume = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=1.0,
                            label="Original Audio Volume",
                            info="Volume of original audio from driving video"
                        )
                    
                    normalize_audio_enabled = gr.Checkbox(
                        value=True,
                        label="Normalize Audio",
                        info="Normalize audio levels for consistent volume"
                    )
                    
                    loop_audio = gr.Checkbox(
                        value=False,
                        label="Loop Audio",
                        info="Loop audio if shorter than video duration"
                    )
                    
                    add_background_music_enabled.change(
                        fn=lambda x: (gr.update(visible=x), gr.update(visible=x)),
                        inputs=[add_background_music_enabled],
                        outputs=[background_music_file, music_settings_row]
                    )
            
            with gr.Row():
                with gr.Column():
                    with gr.Accordion("🔍 Enhancement Settings", open=False):
                        # Model selection
                        if ENHANCED_VIDEO_AVAILABLE and hasattr(enhancer, 'get_available_models'):
                            try:
                                available_models = enhancer.get_available_models()
                                model_choices = [f"{m['name']} - {m['description']}" for m in available_models]
                                model_default = model_choices[0] if model_choices else "RealESRGAN_x4plus_anime_6B - Anime/illustration optimized"
                            except:
                                model_choices = ["RealESRGAN_x4plus_anime_6B - Anime/illustration optimized (Default)"]
                                model_default = model_choices[0]
                        else:
                            model_choices = ["RealESRGAN_x4plus_anime_6B - Anime/illustration optimized (Default)"]
                            model_default = model_choices[0]
                        
                        esrgan_model = gr.Dropdown(
                            choices=model_choices,
                            value=model_default,
                            label="Real-ESRGAN Model",
                            info="Select the enhancement model"
                        )
                        
                        # Quality preset
                        quality_preset = gr.Radio(
                            choices=['low', 'medium', 'high', 'ultra'],
                            value='high',
                            label="Quality Preset",
                            info="Higher quality = larger file size, slower processing"
                        )
                        
                        # Export format
                        export_format = gr.Radio(
                            choices=['mp4', 'webm', 'mov'],
                            value='mp4',
                            label="Export Format",
                            info="MP4 (best compatibility), WebM (smaller), MOV (Apple)"
                        )
                        
                        # Frame rate
                        output_fps = gr.Slider(
                            minimum=15,
                            maximum=60,
                            step=1,
                            value=30,
                            label="Output Frame Rate (FPS)",
                            info="Higher FPS = smoother but larger file"
                        )
                        
                        gr.Markdown("---")
                        gr.Markdown("### 🎨 Smoothing & Denoising", elem_classes=["section-header"])
                        
                        enhance_smoothing = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.0,
                            label="Temporal Smoothing",
                            info="Reduces jitter between frames (0 = off, 1.0 = max)"
                        )
                        
                        enhance_denoise = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.0,
                            label="Denoising Strength",
                            info="Removes noise from frames (0 = off, 1.0 = max)"
                        )
                        
                        enhance_stabilize = gr.Checkbox(
                            value=False,
                            label="Motion Stabilization",
                            info="Reduces camera shake and unwanted motion"
                        )
                    
                    gr.Markdown("---")
                    gr.Markdown("### 💾 Export Options", elem_classes=["section-header"])
                    
                    if EXPORT_UTILS_AVAILABLE:
                        export_custom_resolution_enabled = gr.Checkbox(
                            value=False,
                            label="Custom Resolution Export",
                            info="Export at custom resolution"
                        )
                        
                        with gr.Row(visible=False) as custom_resolution_row:
                            export_width = gr.Number(
                                value=1920,
                                label="Width",
                                minimum=256,
                                maximum=4096,
                                step=64
                            )
                            export_height = gr.Number(
                                value=1080,
                                label="Height",
                                minimum=256,
                                maximum=4096,
                                step=64
                            )
                            export_maintain_aspect = gr.Checkbox(
                                value=True,
                                label="Maintain Aspect Ratio",
                                info="Add padding to maintain aspect ratio"
                            )
                        
                        export_gif_enabled = gr.Checkbox(
                            value=False,
                            label="Export as GIF",
                            info="Create animated GIF version"
                        )
                        
                        with gr.Row(visible=False) as gif_settings_row:
                            gif_fps = gr.Slider(
                                minimum=5,
                                maximum=30,
                                step=1,
                                value=15,
                                label="GIF Frame Rate",
                                info="Lower FPS = smaller file size"
                            )
                            gif_width = gr.Slider(
                                minimum=256,
                                maximum=1024,
                                step=64,
                                value=512,
                                label="GIF Width",
                                info="Width in pixels (height auto-calculated)"
                            )
                        
                        export_frames_enabled = gr.Checkbox(
                            value=False,
                            label="Export Frame Sequence",
                            info="Export individual frames as images"
                        )
                        
                        with gr.Row(visible=False) as frame_settings_row:
                            frame_format = gr.Radio(
                                choices=['png', 'jpg', 'webp'],
                                value='png',
                                label="Frame Format",
                                info="Image format for exported frames"
                            )
                            frame_step = gr.Number(
                                value=1,
                                label="Frame Step",
                                minimum=1,
                                maximum=10,
                                step=1,
                                info="Export every Nth frame (1 = all frames)"
                            )
                        
                        export_custom_resolution_enabled.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=[export_custom_resolution_enabled],
                            outputs=[custom_resolution_row]
                        )
                        
                        export_gif_enabled.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=[export_gif_enabled],
                            outputs=[gif_settings_row]
                        )
                        
                        export_frames_enabled.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=[export_frames_enabled],
                            outputs=[frame_settings_row]
                        )
                    else:
                        export_custom_resolution_enabled = gr.Checkbox(visible=False)
                        export_width = gr.Number(visible=False)
                        export_height = gr.Number(visible=False)
                        export_maintain_aspect = gr.Checkbox(visible=False)
                        export_gif_enabled = gr.Checkbox(visible=False)
                        gif_fps = gr.Slider(visible=False)
                        gif_width = gr.Slider(visible=False)
                        export_frames_enabled = gr.Checkbox(visible=False)
                        frame_format = gr.Radio(visible=False)
                        frame_step = gr.Number(visible=False)
                    
                    enhance_button = gr.Button(
                        "🔍 Enhance with Real-ESRGAN",
                        variant="secondary",
                        size="lg",
                        scale=2
                    )
                    enhanced_video = gr.Video(
                        label="Enhanced Video (Real-ESRGAN)",
                        show_label=True,
                        height=400
                    )
        
        # Retargeting Tab
        with gr.Tab("🎯 Retargeting", id="retargeting_tab"):
            gr.Markdown(
                '<p class="info-text">Adjust the eyes and lip open ratio of the source portrait. Drag the sliders and click Retargeting to apply changes. Try running it multiple times with different values. Set both ratios to 0.8 to see the maximum effect!</p>',
                elem_classes=["info-text"]
            )
            
            with gr.Row():
                eye_retargeting_slider = gr.Slider(
                    minimum=0,
                    maximum=0.8,
                    step=0.01,
                    value=0.0,
                    label="Target Eyes-Open Ratio",
                    info="Control how open the eyes should be (0 = closed, 0.8 = wide open)",
                    show_label=True
                )
                lip_retargeting_slider = gr.Slider(
                    minimum=0,
                    maximum=0.8,
                    step=0.01,
                    value=0.0,
                    label="Target Lip-Open Ratio",
                    info="Control how open the lips should be (0 = closed, 0.8 = wide open)",
                    show_label=True
                )
            
            # Advanced Retargeting Controls
            if RETARGETING_UTILS_AVAILABLE:
                gr.Markdown("---")
                gr.Markdown("### 🎯 Advanced Retargeting Controls", elem_classes=["section-header"])
                
                with gr.Accordion("⚙️ Expression & Movement Controls", open=False):
                    expression_intensity = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        label="Expression Intensity",
                        info="Amplify or reduce expression strength (1.0 = normal, <1.0 = subtle, >1.0 = exaggerated)"
                    )
                    
                    blink_frequency = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        label="Blink Frequency",
                        info="Control blink rate (0.0 = no blinks, 1.0 = normal, 2.0 = frequent)"
                    )
                    
                    head_movement_intensity = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        step=0.1,
                        value=1.0,
                        label="Head Movement Intensity",
                        info="Control head movement amount (0.0 = no movement, 1.0 = normal, 2.0 = exaggerated)"
                    )
                    
                    with gr.Row():
                        gaze_direction_x = gr.Slider(
                            minimum=-1.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.0,
                            label="Gaze Direction X (Left/Right)",
                            info="-1.0 = left, 0.0 = center, 1.0 = right"
                        )
                        gaze_direction_y = gr.Slider(
                            minimum=-1.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.0,
                            label="Gaze Direction Y (Up/Down)",
                            info="-1.0 = down, 0.0 = center, 1.0 = up"
                        )
                
                gr.Markdown("---")
                gr.Markdown("### 🎨 Expression Presets", elem_classes=["section-header"])
                
                expression_preset = gr.Radio(
                    choices=['none'] + ExpressionPresets.list_presets() if RETARGETING_UTILS_AVAILABLE else ['none'],
                    value='none',
                    label="Expression Preset",
                    info="Quick presets for common expressions"
                )
                
                preset_description = gr.Markdown(
                    value="Select a preset to quickly apply expression settings, or use 'none' for manual control.",
                    visible=True
                )
                
                def update_preset_description(preset_name):
                    if preset_name == 'none':
                        return "Select a preset to quickly apply expression settings, or use 'none' for manual control."
                    desc = ExpressionPresets.get_preset_description(preset_name) if RETARGETING_UTILS_AVAILABLE else ""
                    return f"**{preset_name.title()}**: {desc}"
                
                def apply_preset(preset_name):
                    if preset_name == 'none' or not RETARGETING_UTILS_AVAILABLE:
                        return (
                            gr.update(value=0.0),  # eye_open_ratio
                            gr.update(value=0.0),  # lip_open_ratio
                            gr.update(value=1.0),  # expression_intensity
                            gr.update(value=1.0),  # blink_frequency
                            gr.update(value=1.0),  # head_movement_intensity
                            gr.update(value=0.0),  # gaze_direction_x
                            gr.update(value=0.0)   # gaze_direction_y
                        )
                    
                    params = create_retargeting_params_from_preset(preset_name)
                    return (
                        gr.update(value=params.eye_open_ratio),
                        gr.update(value=params.lip_open_ratio),
                        gr.update(value=params.expression_intensity),
                        gr.update(value=params.blink_frequency),
                        gr.update(value=params.head_movement_intensity),
                        gr.update(value=params.gaze_direction_x),
                        gr.update(value=params.gaze_direction_y)
                    )
                
                expression_preset.change(
                    fn=update_preset_description,
                    inputs=[expression_preset],
                    outputs=[preset_description]
                )
                
                expression_preset.change(
                    fn=apply_preset,
                    inputs=[expression_preset],
                    outputs=[
                        eye_retargeting_slider,
                        lip_retargeting_slider,
                        expression_intensity,
                        blink_frequency,
                        head_movement_intensity,
                        gaze_direction_x,
                        gaze_direction_y
                    ]
                )
            else:
                expression_intensity = gr.Slider(visible=False)
                blink_frequency = gr.Slider(visible=False)
                head_movement_intensity = gr.Slider(visible=False)
                gaze_direction_x = gr.Slider(visible=False)
                gaze_direction_y = gr.Slider(visible=False)
                expression_preset = gr.Radio(visible=False)
                preset_description = gr.Markdown(visible=False)
            
            gr.Markdown("---")
            gr.Markdown("### 🖼️ Retargeting Results", elem_classes=["section-header"])
            
            with gr.Row():
                with gr.Column():
                    retargeting_input_image = gr.Image(
                        type="filepath",
                        label="Input Image",
                        show_label=True,
                        height=400
                    )
                    gr.Examples(
                        examples=[
                            [osp.join(example_portrait_dir, "s9.jpg")],
                            [osp.join(example_portrait_dir, "s6.jpg")],
                            [osp.join(example_portrait_dir, "s10.jpg")],
                            [osp.join(example_portrait_dir, "s5.jpg")],
                            [osp.join(example_portrait_dir, "s7.jpg")],
                            [osp.join(example_portrait_dir, "s12.jpg")],
                        ],
                        inputs=[retargeting_input_image],
                        cache_examples=False,
                        label="Example Portraits"
                    )
                with gr.Column():
                    output_image = gr.Image(
                        type="numpy",
                        label="Retargeting Result",
                        show_label=True,
                        height=400
                    )
                with gr.Column():
                    output_image_paste_back = gr.Image(
                        type="numpy",
                        label="Paste-Back Result",
                        show_label=True,
                        height=400
                    )
            
            with gr.Row(elem_classes=["button-group"]):
                process_button_retargeting = gr.Button(
                    "🎗️ Apply Retargeting",
                    variant="primary",
                    size="lg",
                    scale=2
                )
                process_button_reset_retargeting = gr.ClearButton(
                    [
                        eye_retargeting_slider,
                        lip_retargeting_slider,
                        retargeting_input_image,
                        output_image,
                        output_image_paste_back
                    ],
                    value="🧹 Clear",
                    size="lg"
                )
        
        # Batch Processing Tab
        with gr.Tab("📦 Batch Processing", id="batch_tab"):
            gr.Markdown(
                '<p class="info-text">Process multiple images and videos at once. Upload multiple files and they will be processed in a queue.</p>',
                elem_classes=["info-text"]
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Upload Files", elem_classes=["section-header"])
                    batch_image_input = gr.File(
                        file_count="multiple",
                        file_types=["image"],
                        label="Upload Portrait Images",
                        show_label=True
                    )
                    batch_video_input = gr.File(
                        file_count="multiple",
                        file_types=["video"],
                        label="Upload Driving Videos",
                        show_label=True
                    )
                    
                    with gr.Accordion("⚙️ Batch Settings", open=False):
                        batch_relative_motion = gr.Checkbox(
                            value=True,
                            label="Relative Motion",
                            info="Use relative motion for all animations"
                        )
                        batch_do_crop = gr.Checkbox(
                            value=True,
                            label="Crop Source Images",
                            info="Crop portraits before processing"
                        )
                        batch_remap = gr.Checkbox(
                            value=True,
                            label="Paste Back",
                            info="Paste results back into original images"
                        )
                        
                        batch_enhance = gr.Checkbox(
                            value=False,
                            label="Auto-Enhance with Real-ESRGAN",
                            info="Automatically enhance all generated videos"
                        )
                        
                        batch_esrgan_model = gr.Dropdown(
                            choices=model_choices if 'model_choices' in locals() else ["RealESRGAN_x4plus_anime_6B - Anime/illustration optimized (Default)"],
                            value=model_default if 'model_default' in locals() else "RealESRGAN_x4plus_anime_6B - Anime/illustration optimized (Default)",
                            label="Enhancement Model",
                            visible=False
                        )
                        
                        batch_quality = gr.Radio(
                            choices=['low', 'medium', 'high', 'ultra'],
                            value='high',
                            label="Enhancement Quality",
                            visible=False
                        )
                        
                        batch_enhance.change(
                            fn=lambda x: gr.update(visible=x),
                            inputs=[batch_enhance],
                            outputs=[batch_esrgan_model, batch_quality]
                        )
                    
                    with gr.Row(elem_classes=["button-group"]):
                        batch_process_button = gr.Button(
                            "🚀 Start Batch Processing",
                            variant="primary",
                            size="lg",
                            scale=2
                        )
                        batch_clear_button = gr.ClearButton(
                            [batch_image_input, batch_video_input],
                            value="🧹 Clear All",
                            size="lg"
                        )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Processing Status", elem_classes=["section-header"])
                    batch_status = gr.Textbox(
                        label="Queue Status & Progress",
                        value="Ready to process. Upload images and videos, then click 'Start Batch Processing'.",
                        lines=15,
                        interactive=False,
                        show_label=True,
                        elem_classes=["monospace"]
                    )
                    
                    batch_progress = gr.Progress()
                    
                    batch_queue_info = gr.Markdown(
                        value="**Queue System:** Items will be processed sequentially with detailed progress tracking.",
                        visible=True
                    )
                    
                    gr.Markdown("### 📥 Results", elem_classes=["section-header"])
                    batch_results = gr.File(
                        file_count="multiple",
                        label="Download Processed Files",
                        show_label=True,
                        visible=False
                    )
                    
                    batch_download_all = gr.Button(
                        "📦 Download All Results (ZIP)",
                        variant="secondary",
                        size="lg",
                        visible=False
                    )
    
    # Bind functions for buttons
    process_button_retargeting.click(
        fn=gpu_wrapped_execute_image,
        inputs=[eye_retargeting_slider, lip_retargeting_slider, retargeting_input_image, flag_do_crop_input],
        outputs=[output_image, output_image_paste_back],
        show_progress=True,
        api_name="retarget"
    )
    # Face detection
    if FACE_DETECTION_AVAILABLE:
        detect_faces_button.click(
            fn=detect_faces_interface,
            inputs=[image_input],
            outputs=[face_detection_result, face_detection_result, selected_faces],
            show_progress=True,
            api_name="detect_faces"
        )
    
    process_button_animation.click(
        fn=gpu_wrapped_execute_video,
        inputs=[
            image_input,
            video_input,
            flag_relative_input,
            flag_do_crop_input,
            flag_remap_input,
            flag_crop_driving_video_input,
            smoothing_strength,
            denoise_strength,
            stabilize_motion,
            selected_faces,
            aspect_ratio,
            custom_width,
            custom_height,
            crop_mode,
            preserve_bg,
            enable_audio_sync if AUDIO_PROCESSING_AVAILABLE else gr.Checkbox(visible=False),
            add_background_music_enabled if AUDIO_PROCESSING_AVAILABLE else gr.Checkbox(visible=False),
            background_music_file if AUDIO_PROCESSING_AVAILABLE else gr.File(visible=False),
            music_volume if AUDIO_PROCESSING_AVAILABLE else gr.Slider(visible=False),
            original_audio_volume if AUDIO_PROCESSING_AVAILABLE else gr.Slider(visible=False),
            normalize_audio_enabled if AUDIO_PROCESSING_AVAILABLE else gr.Checkbox(visible=False),
            loop_audio if AUDIO_PROCESSING_AVAILABLE else gr.Checkbox(visible=False)
        ],
        outputs=[output_video, output_video_concat],
        show_progress=True,
        api_name="animate"
    )
    enhance_button.click(
        fn=enhance_video_interface,
        inputs=[
            output_video_concat,
            esrgan_model,
            quality_preset,
            export_format,
            output_fps,
            enhance_smoothing,
            enhance_denoise,
            enhance_stabilize,
            export_custom_resolution_enabled if EXPORT_UTILS_AVAILABLE else gr.Checkbox(visible=False),
            export_width if EXPORT_UTILS_AVAILABLE else gr.Number(visible=False),
            export_height if EXPORT_UTILS_AVAILABLE else gr.Number(visible=False),
            export_maintain_aspect if EXPORT_UTILS_AVAILABLE else gr.Checkbox(visible=False),
            export_gif_enabled if EXPORT_UTILS_AVAILABLE else gr.Checkbox(visible=False),
            gif_fps if EXPORT_UTILS_AVAILABLE else gr.Slider(visible=False),
            gif_width if EXPORT_UTILS_AVAILABLE else gr.Slider(visible=False),
            export_frames_enabled if EXPORT_UTILS_AVAILABLE else gr.Checkbox(visible=False),
            frame_format if EXPORT_UTILS_AVAILABLE else gr.Radio(visible=False),
            frame_step if EXPORT_UTILS_AVAILABLE else gr.Number(visible=False)
        ],
        outputs=[enhanced_video],
        show_progress=True,
        api_name="enhance"
    )
    
    # Batch processing with enhanced status updates
    def process_batch_with_updates(*args, progress=gr.Progress()):
        """Wrapper to update status textbox during processing."""
        result = process_batch(*args, progress=progress)
        return result
    
    batch_process_button.click(
        fn=process_batch_with_updates,
        inputs=[
            batch_image_input,
            batch_video_input,
            batch_relative_motion,
            batch_do_crop,
            batch_remap,
            batch_enhance,
            batch_esrgan_model,
            batch_quality
        ],
        outputs=[batch_status, batch_results, batch_results, batch_download_all],
        show_progress=True,
        api_name="batch_process"
    )
    
    def download_batch_zip():
        """Download the latest batch ZIP file."""
        batch_output_dir = live_portrait_output_dir / 'batch_output'
        if batch_output_dir.exists():
            zip_files = sorted(batch_output_dir.glob("batch_results_*.zip"), key=os.path.getmtime, reverse=True)
            if zip_files:
                return zip_files[0]
        return None
    
    batch_download_all.click(
        fn=download_batch_zip,
        outputs=[batch_results],
        api_name="download_batch_zip"
    )
    
    # Update batch results visibility when processing completes
    def update_batch_ui(status_text, results):
        """Update batch UI elements visibility."""
        has_results = results is not None and len(results) > 0 if results else False
        return (
            gr.update(visible=has_results),
            gr.update(visible=has_results)
    )

demo.launch(
    # server_port=args.server_port,
    share=True,
    server_name=args.server_name,
    server_port=8080  # Specify a different port here
)
