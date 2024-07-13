#app.py

# coding: utf-8
# Choose your image
# Choose your Facial Expression (driving video)
# Kick-Starts Multi-Threading Operation on Windows Operating System,
# downscales the original Facial Expression (driving video),
# Enhances frames using Real-Esrgan,
# Enhances Generated Video,
# Re-Assemble Video for Download

import os
import subprocess
from pathlib import Path
import tyro
import gradio as gr
import os.path as osp
from multiprocessing import Pool
from src.utils.helper import load_description
from src.gradio_pipeline import GradioPipeline
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.utils.video import has_audio_stream, exec_cmd

# Paths
project_root = Path('D:/tools/LivePortrait')
live_portrait_output_dir = project_root / 'output'
esrgan_input_dir = live_portrait_output_dir / 'frames'
esrgan_output_dir = live_portrait_output_dir / 'enhanced_frames'
esrgan_model_path = project_root / 'pretrained_weights' / 'RealESRGAN_x4plus_anime_6B.pth'
esrgan_script_path = project_root / 'real-esrgan' / 'inference_realesrgan.py'

# Ensure the output directory exists
os.makedirs(live_portrait_output_dir, exist_ok=True)
os.makedirs(esrgan_input_dir, exist_ok=True)
os.makedirs(esrgan_output_dir, exist_ok=True)


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


def enhance_frame(frame_path):
    command = [
        'python', str(esrgan_script_path), '-n', 'RealESRGAN_x4plus_anime_6B',
        '-i', str(frame_path), '-o', str(esrgan_output_dir / frame_path.name),
        '--model_path', str(esrgan_model_path)
    ]
    subprocess.run(command, check=True)


def enhance_frames(input_dir):
    frame_paths = list(input_dir.glob('*.png'))
    with Pool() as pool:
        pool.map(enhance_frame, frame_paths)


def reassemble_video(input_dir, output_video_path, fps=30):
    command = [
        'ffmpeg', '-framerate', str(fps), '-i',
        str(input_dir / 'frame_%04d.png'), '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p', str(output_video_path)
    ]
    subprocess.run(command, check=True)


def enhance_video(video_path):
    downscaled_video_path = live_portrait_output_dir / 'downscaled_video.mp4'
    enhanced_video_path = live_portrait_output_dir / 'enhanced_live_portrait.mp4'

    # Downscale the video
    downscale_video(video_path, downscaled_video_path)

    # Extract frames from the video
    extract_frames(downscaled_video_path, esrgan_input_dir)

    # Enhance frames using Real-ESRGAN
    enhance_frames(esrgan_input_dir)

    # Reassemble the enhanced frames into a video
    reassemble_video(esrgan_output_dir, enhanced_video_path)

    return enhanced_video_path


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


def gpu_wrapped_execute_video(*args, **kwargs):
    video_path = gradio_pipeline.execute_video(*args, **kwargs)
    return video_path


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

# Define components first
eye_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="Target eyes-open ratio")
lip_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="Target lip-open ratio")
retargeting_input_image = gr.Image(type="filepath")
output_image = gr.Image(type="numpy")
output_image_paste_back = gr.Image(type="numpy")
output_video = gr.Video()
output_video_concat = gr.Video()

# Placeholder for enhanced video output
enhanced_video = gr.Video()


def enhance_video_interface(video_path):
    enhanced_video_path = enhance_video(video_path)
    return enhanced_video_path


with gr.Blocks(theme=gr.themes.Soft(), css=".gradio-container {max-width: 100%; margin: auto;}") as demo:
    gr.Markdown("# LivePortrait based PresentaPulse")
    gr.Markdown("### Efficient Portrait Animation with Smoothing and Retargeting Control")
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Source Portrait")
            image_input = gr.Image(type="filepath")
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
            )
        with gr.Column():
            gr.Markdown("#### Driving Video")
            video_input = gr.Video()
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
            )
    with gr.Row():
        with gr.Accordion(open=False, label="Animation Instructions and Options"):
            with gr.Row():
                flag_relative_input = gr.Checkbox(value=True, label="Relative motion")
                flag_do_crop_input = gr.Checkbox(value=True, label="Do crop (source)")
                flag_remap_input = gr.Checkbox(value=True, label="Paste-back")
                flag_crop_driving_video_input = gr.Checkbox(value=False, label="Do crop (driving video)")
    with gr.Row():
        process_button_animation = gr.Button("🚀 Animate", variant="primary")
        process_button_reset = gr.ClearButton([image_input, video_input, output_video, output_video_concat],
                                              value="🧹 Clear")
        enhance_button = gr.Button("🔍 Enhance with Real-ESRGAN", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### The animated video in the original image space")
            output_video.render()
        with gr.Column():
            gr.Markdown("#### The animated video")
            output_video_concat.render()
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Enhanced video using Real-ESRGAN")
            enhanced_video.render()
    gr.Markdown("### Retargeting")
    gr.Markdown(
        "To edit the eyes and lip open ratio of the source portrait, drag the sliders and click the Retargeting button. You can try running it multiple times. Set both ratios to 0.8 to see what's going on!")
    with gr.Row():
        eye_retargeting_slider.render()
        lip_retargeting_slider.render()
    with gr.Row():
        process_button_retargeting = gr.Button("🚗 Retargeting", variant="primary")
        process_button_reset_retargeting = gr.ClearButton(
            [
                eye_retargeting_slider,
                lip_retargeting_slider,
                retargeting_input_image,
                output_image,
                output_image_paste_back
            ],
            value="🧹 Clear"
        )
    with gr.Row():
        with gr.Column():
            gr.Markdown("#### Retargeting Input")
            retargeting_input_image.render()
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
            )
        with gr.Column():
            gr.Markdown("#### Retargeting Result")
            output_image.render()
        with gr.Column():
            gr.Markdown("#### Paste-back Result")
            output_image_paste_back.render()
    # binding functions for buttons
    process_button_retargeting.click(
        fn=gpu_wrapped_execute_image,
        inputs=[eye_retargeting_slider, lip_retargeting_slider, retargeting_input_image, flag_do_crop_input],
        outputs=[output_image, output_image_paste_back],
        show_progress=True
    )
    process_button_animation.click(
        fn=gpu_wrapped_execute_video,
        inputs=[
            image_input,
            video_input,
            flag_relative_input,
            flag_do_crop_input,
            flag_remap_input,
            flag_crop_driving_video_input
        ],
        outputs=[output_video, output_video_concat],
        show_progress=True
    )
    enhance_button.click(
        fn=enhance_video_interface,
        inputs=[output_video_concat],
        outputs=[enhanced_video],
        show_progress=True
    )

demo.launch(
    # server_port=args.server_port,
    share=True,
    server_name=args.server_name,
    server_port=8080  # Specify a different port here
)
