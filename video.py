import subprocess
import logging
from tqdm import tqdm
from multiprocessing import Pool

# Full path to ffprobe
FFPROBE_PATH = 'C:\\ffmpeg\\bin\\ffprobe.exe'

def exec_cmd(cmd, total_steps=None):
    try:
        with tqdm(total=total_steps, desc="Processing", unit="step") as pbar:
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            while True:
                output = process.stdout.readline()
                if process.poll() is not None and output == b'':
                    break
                if output:
                    pbar.update(1)
                    logging.info(output.strip().decode('utf-8'))
            result = process.communicate()[0]
            return result.decode('utf-8')
    except subprocess.CalledProcessError as e:
        logging.error(f"Command '{cmd}' failed with error: {e.stderr.decode('utf-8')}")
        raise

def has_audio_stream(video_path):
    cmd = [FFPROBE_PATH, '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
    try:
        result = exec_cmd(' '.join(cmd))
        return 'audio' in result
    except Exception as e:
        logging.error(f"Error occurred while probing video: {video_path}, you may need to install ffprobe! Now set audio to false!")
        return False

def enhance_frame(frame_path, esrgan_script_path, esrgan_model_path, esrgan_output_dir):
    command = [
        'python', str(esrgan_script_path), '-n', 'RealESRGAN_x4plus_anime_6B',
        '-i', str(frame_path), '-o', str(esrgan_output_dir / frame_path.name),
        '--model_path', str(esrgan_model_path)
    ]
    subprocess.run(command, check=True)

class VideoEnhancer:
    def __init__(self, esrgan_script_path, esrgan_model_path, esrgan_output_dir):
        self.esrgan_script_path = esrgan_script_path
        self.esrgan_model_path = esrgan_model_path
        self.esrgan_output_dir = esrgan_output_dir

    def enhance_frames(self, input_dir):
        frame_paths = list(input_dir.glob('*.png'))
        with Pool() as pool:
            pool.starmap(enhance_frame, [(frame, self.esrgan_script_path, self.esrgan_model_path, self.esrgan_output_dir) for frame in frame_paths])
