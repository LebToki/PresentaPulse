class GradioPipeline:
    def __init__(self, inference_cfg=None, crop_cfg=None, args=None):
        pass
    def execute_video(self, image_path, video_path, relative_motion, do_crop, remap, crop_driving_video):
        return ("output",)
    def execute_image(self, *args, **kwargs):
        return ("image",)
