from typing import List
import os
from setuptools import find_packages
from setuptools import setup
import glob
from typing import List
import shutil
import tqdm
import cv2
import tempfile
import numpy as np
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.engine.defaults import DefaultPredictor


class Test:
    def __init__(self):
        self.cfg = get_cfg()
        self.model = 'faster_rcnn_R_50_C4_1x.yaml'
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/" + self.model))
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/" + self.model)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        self.cfg.MODEL.DEVICE = "cpu"
        self.instance_mode=ColorMode.IMAGE
        self.video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)
        self.predictor = DefaultPredictor(self.cfg)


        '''def get_model_zoo_configs(self) -> List[str]:
            """
            Return a list of configs to include in package for model zoo. Copy over these configs inside
            detectron2/model_zoo.
            """

            # Use absolute paths while symlinking.
            source_configs_dir = path.join(path.dirname(path.realpath(__file__)), "configs")
            destination = path.join(
                path.dirname(path.realpath(__file__)), "detectron2", "model_zoo", "configs"
            )
            print('src:', source_configs_dir)
            print('dest:', destination)
            # Symlink the config directory inside package to have a cleaner pip install.

            # Remove stale symlink/directory from a previous build.
            if path.exists(source_configs_dir):
                print('In first if')
                if path.islink(destination):
                    path.unlink(destination)
                elif path.isdir(destination):
                    shutil.rmtree(destination)

            if not path.exists(destination):
                try:
                    print('in try block')
                    path.symlink(source_configs_dir, destination)
                except OSError:
                    # Fall back to copying if symlink fails: ex. on Windows.
                    shutil.copytree(source_configs_dir, destination)

            config_paths = glob.glob(destination + '/*.yaml', recursive=True) + glob.glob(
                destination + '/*.py', recursive=True
            )
            config_paths = glob.glob("configs/**/*.yaml", recursive=True) + glob.glob(
                "configs/**/*.py", recursive=True
            )
            print(config_paths)
            return config_paths'''

    def test_opencv_video_format(self, codec, file_ext):
        with tempfile.TemporaryDirectory(prefix="video_format_test") as dir:
            filename = os.path.join(dir, "test_file" + file_ext)
            writer = cv2.VideoWriter(
                filename=filename,
                fourcc=cv2.VideoWriter_fourcc(*codec),
                fps=float(30),
                frameSize=(10, 10),
                isColor=True,
            )
            [writer.write(np.zeros((10, 10, 3), np.uint8)) for _ in range(30)]
            writer.release()
            if os.path.isfile(filename):
                return True
            return False

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def detectron2_pred_video(self, video_path):
        video = cv2.VideoCapture(video_path)
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        print('num of frames:', num_frames)
        print('frame_height, frame_width:', width, height)

        # codec, file_ext = (('x264', '.mkv') if self.test_opencv_video_format('x264', '.mkv') else ('.mp4v', '.mp4'))
        codec, file_ext = ('mp4v', '.mp4')
        base_name = 'pred_' + os.path.basename(video_path)
        output_fname = os.path.splitext(base_name)[0] + file_ext
        print('name:', output_fname)

        output_file = cv2.VideoWriter(
            filename=output_fname,
            # some installation of opencv may not support x264 (due to its license),
            # you can try other format (e.g. MPEG)
            fourcc=cv2.VideoWriter_fourcc(*codec),
            fps=float(frames_per_second),
            frameSize=(int(width), int(height)),
            isColor=True,
        )
        frame_gen = self._frame_from_video(video)

        #print('frame_gen:', len(frame_gen))

        for frame in tqdm.tqdm(frame_gen, total = num_frames):
            output = self.predictor(frame)
            vis_frame = self.video_visualizer.draw_instance_predictions(frame, output['instances'].to('cpu'))
            output_file.write(vis_frame.get_image())

        output_file.release()
        video.release()

    def yolo_pred_video(self):



if __name__ == "__main__":
    tst = Test()
    #data = {"detectron2.model_zoo": tst.get_model_zoo_configs()}
    #print(data)
    tst.pred_video('video1.mp4')

