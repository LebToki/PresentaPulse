# coding: utf-8

import os
import subprocess
from pathlib import Path
import tyro
import gradio as gr
import os.path as osp
from src.utils.helper import load_description
from src.gradio_pipeline import GradioPipeline
from src.config.crop_config import CropConfig
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.utils.video import has_audio_stream, exec_cmd, VideoEnhancer

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

enhancer = VideoEnhancer(esrgan_script_path, esrgan_model_path, esrgan_output_dir)


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


def enhance_video(video_path):
    downscaled_video_path = live_portrait_output_dir / 'downscaled_video.mp4'
    enhanced_video_path = live_portrait_output_dir / 'enhanced_live_portrait.mp4'

    # Downscale the video
    downscale_video(video_path, downscaled_video_path)

    # Extract frames from the video
    extract_frames(downscaled_video_path, esrgan_input_dir)

    # Enhance frames using Real-ESRGAN
    enhancer.enhance_frames(esrgan_input_dir)

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

def enhance_video_interface(video_path):
    enhanced_video_path = enhance_video(video_path)
    return enhanced_video_path


# Modern CSS styling
custom_css = """
    .gradio-container {
        max-width: 1400px !important;
        margin: auto;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    .section-header {
        font-weight: 600;
        color: #1e293b;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }
    .info-text {
        background: #f1f5f9;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #475569;
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
        border-top: 2px solid #e2e8f0;
    }
"""

# Create modern theme
theme = gr.themes.Monochrome(
    primary_hue="purple",
    secondary_hue="slate",
    font=("Inter", "ui-sans-serif", "system-ui", "sans-serif"),
).set(
    button_primary_background_fill="#667eea",
    button_primary_background_fill_hover="#5568d3",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#f1f5f9",
    button_secondary_background_fill_hover="#e2e8f0",
    border_color_accent="#667eea",
    shadow_drop_lg="0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)",
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
            
            with gr.Row():
                with gr.Column():
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
    # Bind functions for buttons
    process_button_retargeting.click(
        fn=gpu_wrapped_execute_image,
        inputs=[eye_retargeting_slider, lip_retargeting_slider, retargeting_input_image, flag_do_crop_input],
        outputs=[output_image, output_image_paste_back],
        show_progress=True,
        api_name="retarget"
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
        show_progress=True,
        api_name="animate"
    )
    enhance_button.click(
        fn=enhance_video_interface,
        inputs=[output_video_concat],
        outputs=[enhanced_video],
        show_progress=True,
        api_name="enhance"
    )

demo.launch(
    # server_port=args.server_port,
    share=True,
    server_name=args.server_name,
    server_port=8080  # Specify a different port here
)
