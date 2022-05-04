from typing import List
import os
from setuptools import find_packages
from setuptools import setup
import glob
from typing import List
import logging
import tqdm
import cv2
import tempfile
import torch
import numpy as np
from numpy import random
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.engine.defaults import DefaultPredictor

#from predictor_yolo_detector.utils.augmentations import letterbox
from predictor_yolo_detector.utils.general import (check_img_size, non_max_suppression, 
                                                scale_coords, plot_one_box)

log_str = "[%(asctime)s: %(levelname)s: %(module)s] %(message)s"
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_dir, "test_log.log"), level=logging.INFO, format=log_str, datefmt='%Y:%m:%d %H:%M:%S')

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
        self.detectron2_predictor = DefaultPredictor(self.cfg)

        self.conf = float(0.3)
        self.view_img = False
        self.save_txt = False
        self.device = 'cpu'
        self.augment = True
        self.agnostic_nms = True
        self.conf_thres = float(0.3)
        self.iou_thres = float(0.45)
        self.save_conf = True
        self.update = True
        self.line_thickness = 3
        self.yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

        


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
            logging.info('src:', source_configs_dir)
            logging.info('dest:', destination)
            # Symlink the config directory inside package to have a cleaner pip install.

            # Remove stale symlink/directory from a previous build.
            if path.exists(source_configs_dir):
                logging.info('In first if')
                if path.islink(destination):
                    path.unlink(destination)
                elif path.isdir(destination):
                    shutil.rmtree(destination)

            if not path.exists(destination):
                try:
                    logging.info('in try block')
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
            logging.info(config_paths)
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

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)


    def detectron2_pred_video(self, video_path):
        video = cv2.VideoCapture(video_path)
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        logging.info('num of frames:', num_frames)
        logging.info('frame_height, frame_width:', width, height)

        # codec, file_ext = (('x264', '.mkv') if self.test_opencv_video_format('x264', '.mkv') else ('.mp4v', '.mp4'))
        codec, file_ext = ('mp4v', '.mp4')
        base_name = 'pred_' + os.path.basename(video_path)
        output_fname = os.path.splitext(base_name)[0] + file_ext
        logging.info('name:', output_fname)

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

        logging.info(f'frame_gen:{len(frame_gen)}')

        for frame in tqdm.tqdm(frame_gen, total = num_frames):
            output = self.detectron2_predictor(frame)
            vis_frame = self.video_visualizer.draw_instance_predictions(frame, output['instances'].to('cpu'))
            output_file.write(vis_frame.get_image())

        output_file.release()
        video.release()

    def yolo_pred_video(self, video_path):
        video = cv2.VideoCapture(video_path)
        width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

        codec, file_ext = ('mp4v', '.mp4')
        base_name = 'pred1_' + os.path.basename(video_path)
        output_fname = os.path.splitext(base_name)[0] + file_ext

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

        stride, names, pt = self.yolo_model.stride, self.yolo_model.names, self.yolo_model.pt
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        for frame in tqdm.tqdm(frame_gen, total=num_frames):
            framesz = list((int(height), int(width)))
            framesz = [check_img_size(x, s=stride) for x in framesz]
            logging.info(f'pt:{pt}')
            logging.info('framesz:', framesz)
            logging.info('frame type:', type(frame))
            logging.info('frame shape:', frame.shape)
            frame0 = self.letterbox(img=frame, new_shape=framesz, auto=pt)[0]
            frame0 = torch.from_numpy(frame0).to(self.device)
            frame0 = frame0.float()
            frame0 /= 255.0
            if frame0.ndimension() == 3:
                frame0 = frame0.unsqueeze(0)
            frame0 = frame0.permute(0,3,1,2)
            logging.info('frame0 shape:', frame0.shape)

            pred = self.yolo_model(frame0, augment=self.augment)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres,
                                       agnostic=self.agnostic_nms)

            for i, det in enumerate(pred):
                gn = torch.tensor(frame.shape)[[1, 0, 1, 0]]
                if det is not None and len(det):
                    # Rescale from frame0 size to frame size
                    det[:, :4] = scale_coords(frame0.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (names[int(cls)], conf)
                        plot_one_box(xyxy, frame, label=label, color=colors[int(cls)], 
                                    line_thickness=self.line_thickness)
            
            output_file.write(frame)

        output_file.release()
        video.release()

    def log_statements(self):
        logging.info(f"model:{self.model}")
        logging.info(f"img:{self.view_img}")


if __name__ == "__main__":
    tst = Test()
    #data = {"detectron2.model_zoo": tst.get_model_zoo_configs()}
    #logging.info(data)
    #tst.pred_video('video1.mp4')
    #tst.yolo_pred_video('video1.mp4')
    tst.log_statements()

