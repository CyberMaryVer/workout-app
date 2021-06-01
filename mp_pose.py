import cv2
import multiprocessing as mp
import tqdm
from time import time
import numpy as np
import argparse
from visualization import get_updated_keypoint_dict, draw_joints, draw_skeleton, visualize_keypoints, \
    draw_box_with_text
from mp_predictor import MpipePredictor
from utils import save_results_to_csv


def get_parser():
    parser = argparse.ArgumentParser(description="Mediapipe")

    # MODEL CONFIGURATION
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
    parser.add_argument("--skeleton", default=1, type=int, help="By default draws full skeleton [1], choose [0] "
                                                                "if you don't want to draw it and [2] to draw "
                                                                "only body connections.")
    parser.add_argument("--joints", default=1, type=int, help="By default draws all joints [1], choose [0] to disable")
    parser.add_argument("--side", default=None, help="If [L, R] is choosen, visualizes only one side.")
    parser.add_argument("--mode", default=None, help="A file or directory to save output.")
    parser.add_argument("--instance", default=0, type=int, help="Number of instance for multiperson mode.")
    parser.add_argument("--scale", default=0.5, type=float, help="Set scale parameter for video output.")

    # DEBUG OPTIONS
    parser.add_argument("--verbose", action="store_true", help="Set verbose parameter")
    parser.add_argument("--debug_video", default=None, nargs="+",
                        help="A list of space separated parameters - step and number of frames to process.")

    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    print(args)

    # choose inference parameters
    THRESHOLD = args.threshold

    # set paths to video and photo
    _IMAGE = args.image
    _VIDEO = args.video
    _WEBCAM = args.webcam
    _TEST = args.test

    # settings for the test script
    _SAVE_OUTPUT = args.output
    _SAVE_FILE_NAME = args.output_name
    _DEBUG_VIDEO = args.debug_video
    _DEBUG_PATH_TO_VIDEO = None

    _TRACKING_DEBUG = True

    # visualisation parameters
    _INSTANCE_NUM = args.instance
    _CONNECTIONS = args.skeleton
    _JOINTS = args.joints
    _SIDE = args.side
    _MODE = args.mode  # ["keypoints_names", "angles", "symmetry", "gravity_center"]
    _CSV = args.csv
    _SCALE = args.scale

    predictor = MpipePredictor(detection_thr=THRESHOLD)

    if args.debug_video is not None:
        debug_params = [int(x) for x in args.debug_video]
    else:
        debug_params = None

    if _TEST is not None and _VIDEO is not None:
        _DEBUG_PATH_TO_VIDEO = _VIDEO
        _VIDEO = None

    if _IMAGE is not None and _VIDEO is not None:
        print("You indicated both --image and --video parameter. Inference will be done only for image. "
              "Please run inference on video separately")

    # test outputs and draw custom prediction
    if _IMAGE is not None:

        if _TEST:
            im = cv2.imread(_IMAGE)
            im_wk = np.asarray(im.copy())
            start_time = time()
            outputs = predictor.get_keypoints(im)
            output_name = _IMAGE[:-4] + '_inference_out.jpg'

            kps = get_updated_keypoint_dict(outputs)
            im = draw_skeleton(im, keypoints=kps, threshold=.4, headless=True)
            im, _ = draw_joints(im, keypoints=kps, threshold=.4, headless=True)
            end_time = time()
            time_txt = f'Time of inference = {(end_time - start_time):.2f} sec'
            print(time_txt)
            output_image = draw_box_with_text(im, time_txt)
            # output_image = draw_text(im, time_txt)

        else:
            im = cv2.imread(_IMAGE)
            im_wk = np.asarray(im.copy())
            scale = max(im_wk.shape) / 600
            outputs = outputs = predictor.get_keypoints(im)
            output_name = _IMAGE[:-4] + '_visualisation_out.jpg'

            kps = get_updated_keypoint_dict(outputs)
            output_image = visualize_keypoints(kps, im, skeleton=_CONNECTIONS, dict_is_updated=True,
                                               threshold=.4, side=_SIDE, mode=_MODE, scale=_SCALE, joints=_JOINTS)

        cv2.imshow(output_name, output_image)
        cv2.waitKey(0)
        cv2.imwrite(output_name, output_image)

    elif _VIDEO is not None:
        save_path = _SAVE_FILE_NAME if _SAVE_FILE_NAME is not None else None

        video = MpipePredictor(path_to_video=_VIDEO, detection_thr=THRESHOLD, tracking_thr=.59)
        total = video.num_frames if debug_params is None else debug_params[1]
        for vis_frame, _ in tqdm.tqdm(video.run_on_video(debug_params=debug_params,
                                                         side=_SIDE,
                                                         skeleton=_CONNECTIONS,
                                                         mode=_MODE,
                                                         save_output=_SAVE_OUTPUT,
                                                         save_path=save_path),
                                      total=total):

            cv2.namedWindow(video.basename, cv2.WINDOW_AUTOSIZE)
            video_width, video_height = int(video.width * _SCALE), int(video.height * _SCALE)
            generated_frame = cv2.resize(vis_frame, (video_width, video_height))
            cv2.imshow(video.basename, generated_frame)
            if cv2.waitKey(1) == 27:
                break  # esc to quit

        if _CSV:
            results = video.tracking
            save_results_to_csv(results)

        # if _TRACKING_DEBUG:
        #     import pickle as pk
        #     with open("tracking.pickle", "wb") as f:
        #         pk.dump(video.tracking, f)
        #     print(len(video.tracking), video.tracking[0])

    elif _TEST:
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        _SCALE = 1.4

        while cv2.waitKey(1) != 27:
            ret, frame = cam.read()

            # print(cam.get(3), cam.get(4), cam.get(1))
            try:
                outputs = predictor.get_keypoints(frame)
                kps = get_updated_keypoint_dict(outputs)
                frame = visualize_keypoints(kps, frame, skeleton=_CONNECTIONS, dict_is_updated=True,
                                            threshold=.8, side=_SIDE, mode=_MODE, scale=_SCALE, joints=_JOINTS)
            except Exception as e:
                # print(e)
                pass

            cv2.imshow('frame', frame)

        cam.release()
        cv2.destroyAllWindows()

    else:
        print("Run module with --help parameter to learn about the usage")
