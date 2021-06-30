# -*- coding: utf-8 -*-
# Detectron2 implementation

import argparse
from detectron2 import model_zoo
from detectron2.config import get_cfg
import yaml

PATHS = ["keypoint_rcnn_R_50_FPN_1x.yaml",
         "keypoint_rcnn_R_50_FPN_3x.yaml",
         "keypoint_rcnn_R_101_FPN_3x.yaml",
         "keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"]


def build_cfg(model: str, device: str, thresh=.5):
    """
    creates a fresh new config and builds the model
    """
    cfg = get_cfg()

    config_path = "COCO-Keypoints/" + model
    print(f"load model from {config_path}...")
    cfg.merge_from_file(model_zoo.get_config_file(config_path))

    # choose configuration
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh  # set threshold for this unipose
    # cfg.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"
    # cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
    # cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 0
    # cfg.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = tuple(512 for _ in range(8))
    # cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17  # 17 is the number of keypoints in COCO.
    # cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1
    # cfg.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = True

    # Multi-task loss weight to use for keypoints
    # Recommended values:
    #   - use 1.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is True
    #   - use 4.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is False
    # cfg.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 1.0

    # Type of pooling operation applied to the incoming feature map for each RoI
    # cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE = "ROIAlignV2"

    print("load weights...")
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_path)

    if device == "cpu":
        cfg["MODEL"]["DEVICE"] = device
        print(f'configuration was changed to: {cfg["MODEL"]["DEVICE"]}')

    return cfg


def read_cfg_file(cfg_file):
    """
    read configuration yaml file
    """
    with open(cfg_file) as info:
        info_dict = yaml.load(info, Loader=yaml.FullLoader)
    return info_dict


def build_model_from_cfg(cfg_file):
    """
    receives configuration file and returns the unipose config
    """
    cfg_dict = read_cfg_file(cfg_file)
    model = cfg_dict["MODEL"]
    device = cfg_dict["DEVICE"]
    cfg = build_cfg(model, device)
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2")

    # MODEL CONFIGURATION
    parser.add_argument("--config-file", default=0, type=int, help="config file: [keypoint_rcnn_R_50_FPN_1x.yaml, "
                                                                   "keypoint_rcnn_R_50_FPN_3x.yaml, "
                                                                   "keypoint_rcnn_R_101_FPN_3x.yaml, "
                                                                   "keypoint_rcnn_X_101_32x8d_FPN_3x.yaml]")
    parser.add_argument("--device", default="cuda", help="Type of inference: [cpu, cuda].")
    parser.add_argument("--threshold", type=float, default=0.7, help="Threshold value.")

    # INFERENCE PARAMETERS
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video", help="Path to video file.")
    parser.add_argument("--image", help="Path to image file.")
    parser.add_argument("--output", action="store_true", help="Set this parameter to save the output")
    parser.add_argument("--output_name", default=None, type=str, help="Output file name")
    parser.add_argument("--csv", action="store_true", help="Save results to csv file.")
    parser.add_argument("--test", action="store_true", help="Run test inference on image.")

    # VISUALISATION SETTINGS
    parser.add_argument("--skeleton", default=1, type=int, help="If set shows only keypoints.")
    parser.add_argument("--joints", default=1, type=int, help="If set to 0 shows only connections.")
    parser.add_argument("--side", default=None, help="If [L, R] is choosen, visualizes only one side.")
    parser.add_argument("--mode", default=None, help="Mode:[keypoints_names, angles, symmetry].")
    parser.add_argument("--instance", default=0, type=int, help="Number of instance for multiperson mode.")
    parser.add_argument("--scale", default=0.5, type=float, help="Set scale parameter for video output.")

    # DEBUG OPTIONS
    parser.add_argument("--verbose", action="store_true", help="Set verbose parameter")
    parser.add_argument("--debug_video", default=None, nargs="+",
                        help="A list of space separated parameters - step and number of frames to process.")

    # OTHER
    parser.add_argument('--exercise', default=None, type=str, help='Exercise to perform: ["left_heel_slides", '
                                                                   '"seated_right_knee_extension", '
                                                                   '"side_lying_left_leg_lift",'
                                                                   '"coord_angles",'
                                                                   '"symmetry"].')

    return parser


if __name__ == "__main__":
    # test()
    pass
