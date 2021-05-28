# -*- coding: utf-8 -*-
# Detectron2 & Mediapipe implementation

import os
import cv2
import torch
from inference_config import build_cfg
from inference_config import PATHS
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from geometry import get_central_points
from predictor import AsyncPredictor
from visualization import visualize_keypoints

_KEYPOINT_THRESHOLD = 0.05

INSTANCE_NUM = 0
KEYPOINT_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]


def convert_tensor_to_cords(tensor_list):
    list_from_tensor = tensor_list.flatten().detach().tolist()
    return list_from_tensor


def convert_dict(tensor_dict):
    """
    returns converted dict
    """
    converted_dict = {}
    for keypoint_name, keypoint_coords in tensor_dict.items():
        try:
            converted_item = {keypoint_name: [coord.flatten().tolist()[0] for coord in keypoint_coords]}
            converted_dict.update(converted_item)
        except:
            converted_item = {keypoint_name: [coord for coord in keypoint_coords]}
            converted_dict.update(converted_item)
    return converted_dict


def get_updated_keypoints_from_tensors(keypoint_dict):
    keypoint_dict_ = convert_dict(keypoint_dict)
    points_to_connect = get_central_points(keypoint_dict)
    keypoint_dict_.update(points_to_connect)
    return keypoint_dict_


def create_keypoints_dictionary(outputs, cfg, instance=INSTANCE_NUM):
    """
    creates dictionary {keypoint name: keypoint coords}
    """
    keypoint_dict = {}
    keypoint_names = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).keypoint_names
    if len(outputs["instances"].pred_keypoints) == 0:
        return None
    try:
        keypoint_coords = outputs["instances"].pred_keypoints[instance]
    except IndexError:
        keypoint_coords = outputs["instances"].pred_keypoints[0]
    except Exception as e:
        print(e)
        return None

    for i in zip(keypoint_names, keypoint_coords):
        cords = convert_tensor_to_cords(i[1][:3, ])
        keypoint_dict.update({i[0]: cords})

    return keypoint_dict


def test_from_file(path="instances_predictions.pth"):
    pred = torch.load(path)["instances"].to("cpu").pred_keypoints[INSTANCE_NUM]
    for K in KEYPOINT_NAMES:
        idx = KEYPOINT_NAMES.index(K)
        print(K, pred[idx].flatten().tolist())


def extract_predictions(predictions):
    pred = predictions["instances"].to("cpu").pred_keypoints[INSTANCE_NUM]
    extracted = {}
    for K in KEYPOINT_NAMES:
        idx = KEYPOINT_NAMES.index(K)
        extracted.update({K: pred[idx].flatten().tolist()})
    return extracted


class DtPredictor:
    def __init__(self, device, model_number=1, thresh=.9, parallel=False):
        self.cfg = build_cfg(model=PATHS[model_number], device=device, thresh=thresh)
        self.predictor = DefaultPredictor(self.cfg)
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(self.cfg, num_gpus=num_gpu)

    def get_kps(self, img, instance=0):
        outputs = self.predictor(img)
        kps = create_keypoints_dictionary(outputs, self.cfg, instance)
        kps = convert_dict(kps)
        return kps


class DetectronVideo:
    def __init__(self, path_to_video, cfg, parallel=False, instance=0):
        self.instance = instance

        # general parameters
        self.cfg = cfg
        self.video = cv2.VideoCapture(path_to_video)
        self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale = max(self.height, self.width) / 600
        self.frames_per_second = self.video.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.basename = os.path.basename(path_to_video)
        self.tracking = {}
        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

        # debug parameters
        self.drop_frame_interval = 1
        self.num_frames_for_debug = self.num_frames

    def _frame_from_video(self, video):
        # while video.isOpened():
        f = 0
        while f < self.num_frames_for_debug:
            success, frame = video.read()
            if success:
                yield frame
                f += 1
            else:
                break

    def run_on_video(self, debug_params=None, side=None, skeleton=True, joints=True, mode=None,
                     save_output=False, save_path=None):
        output_file = None
        if save_output:
            if save_path is None:
                output_filename = os.path.join(self.basename.split('.')[0] + "_out.mp4")
            else:
                output_filename = save_path

            output_file = cv2.VideoWriter(
                filename=output_filename,
                # some installation of opencv may not support mp4v (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=float(self.frames_per_second),
                frameSize=(self.width, self.height),
                isColor=True,
            )

        if debug_params is not None:
            print(f"::Debug mode::\ncalculation step = {debug_params[0]}"
                  f"\nnumber of frames = {debug_params[1]}")
            self.drop_frame_interval = debug_params[0]
            self.num_frames_for_debug = debug_params[1]

        frame_gen = self._frame_from_video(self.video)
        current_frame = 0

        read_one_frame = True
        while read_one_frame:
            _, initial_frame = self.video.read()
            read_one_frame = False

        keypoint_dict = None

        for frame in frame_gen:

            if current_frame % self.drop_frame_interval == 0:
                # start = time.time()
                pred = self.predictor(frame)
                # with PathManager.open("instances_predictions.pth", "ab") as f:
                #     torch.save(pred, f)
                keypoint_dict = create_keypoints_dictionary(pred, self.cfg, self.instance)
                # print(f"\ntime (pred): {time.time() - start}")
                self.tracking.update({current_frame: keypoint_dict})

            frame = visualize_keypoints(keypoint_dict, frame, skeleton=skeleton, joints=joints, side=side, mode=mode,
                                        scale=self.scale)

            if save_output:
                output_file.write(frame)

            yield frame, keypoint_dict

            current_frame += 1

        self.video.release()
        if save_output:
            output_file.release()
        else:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    pass
