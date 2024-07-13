# LivePortrait-PresentaPulse

PresentaPulse leverages the power of LivePortrait and Real-ESRGAN to create ultra-realistic animated portraits. Developed by Tarek Tarabichi from [2TInteractive](https://2tinteractive.com), this tool integrates advanced AI models to enhance image quality and create stunning animations.

## Features

- **Realistic Animations**: Generate ultra-realistic animations from still images.
- **Image Enhancement**: Use Real-ESRGAN to upscale and enhance image quality.
- **Interactive UI**: User-friendly interface powered by Gradio.
- **Multiprocessing**: Optimized for performance with multiprocessing support.

![image](https://github.com/user-attachments/assets/913378a1-406d-4a63-b00d-1f1ef3426ff7)

## What to expect (Magic!)
![original](https://github.com/user-attachments/assets/79297188-24dc-4841-83f8-decaf9d67f0a)
![Original with Expression Applied](https://github.com/user-attachments/assets/da6dcde1-7772-4356-bcf5-5b74a8cbf4c4)
![LivePortrait Face Markers is magical](https://github.com/user-attachments/assets/fcb28cb4-f519-4aa5-b7eb-68d655394666)


## How to use:
- Choose your image
- Choose your Facial Expression (driving video)
- Kick-Starts Multi-Threading Operation on Windows Operating System,
- downscales the original Facial Expression (driving video),
- Enhances frames using Real-Esrgan,
- Enhances Generated Video,
- Re-Assemble Video for Download
  
## Installation

### Prerequisites

- Python 3.8 or higher
- Pip
- Virtualenv

### Clone the Repository

```sh
git clone https://github.com/LebToki/PresentaPulse.git
cd PresentaPulse
```

### Create and Activate Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### Install Dependencies
```sh
pip install -r requirements.txt
```

## Usage

### Run the Application
```sh
python app.py
```
Once the application is running, you can access the web interface using the local URL provided in the terminal.

### Recommended Models

**Worth mentioning:**
- You need to mind that I am hardcoding my directory path for now! So, you are recommended to change that path to work with your own system deployment path until I get to a mature version.
- I suggest you clone Real-ESRGAN in the root folder of the project and build it within to avoid messing up with the codebase and compatibility.
- Make sure your Torch has CUDA support.
- Make sure your Onnxruntime has CUDA support.
- Experiment with different expressions (Pexels has a few solid good ones).
- Avoid facial expressions with tons of camera movements, pans, and tilts.
- Have fun and experiment.
- I would appreciate it if you follow me, star the repo, and check the other AI-related repos too :)

**Pretrained Models and Configuration:**
create this directory and download the models from 
```text
D:\tools\LivePortrait\pretrained_weights
```

# you may need to run `git lfs install` first
```text
git clone https://huggingface.co/KwaiVGI/liveportrait pretrained_weights
```
alternatively you may try to download the archive from Google Drive
(https://drive.google.com/drive/folders/1UtKgzKjFAOmZkhNK-OYT0caJ_w2XAnib)

## Directory Structure

```plaintext
realesr-animevideov3.pth
RealESRGAN_x4plus.pth
RealESRGAN_x4plus_anime_6B.pth
RealESRNet_x4plus.pth

├───insightface
│   └───models
│       └───buffalo_l
│               2d106det.onnx
│               det_10g.onnx
│               
└───liveportrait
    │   landmark.onnx
    │   
    ├───base_models
    │       appearance_feature_extractor.pth
    │       motion_extractor.pth
    │       spade_generator.pth
    │       warping_module.pth
    │       
    └───retargeting_models
            stitching_retargeting_module.pth
```

## Acknowledgements

Let'us all give kudos to the original creator of LivePortrait and every other library we are using, including:

- LivePortrait (https://github.com/ymuhong/LivePortrait-Advanced)
- Real-ESRGAN (https://github.com/xinntao/Real-ESRGAN)

## Contributing

I welcome contributions! Please fork the repository and create a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For more information, please visit 2TInteractive (https://2tinteractive.com) or contact Tarek Tarabichi.


## TO DO

- **The Fun-part is still being worked on:** Taking this whole thing into Real-ESRGAN and applying the highest quality possible using `RealESRGAN_x4plus_anime_6B.pth` or `RealESRGAN_x4plus.pth`.
- **The Wishlist is growing:**
  - Possibly integrate multiface with object and zones detection and targeting.
  - Possibly working around the square generation limitation (1:1).
  - Possibly trying to integrate MimicMotion which will require updating the workflow to include the use of MimicMotion for generating motion sequences before or after the Real-ESRGAN enhancement. The only issue is its high demands on PC resources which could be an issue (for now at least).


