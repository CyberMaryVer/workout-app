import csv
import yaml

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


def create_csv_file(filename: str = "results.csv"):
    """
    creates csv file with keypoints names as column names
    """
    keypoint_names = KEYPOINT_NAMES
    with open(filename, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["FRAME#", *keypoint_names, ';'])


def save_results_to_csv(results: dict, new_file: bool = True, filename: str = None):
    """
    saves results in csv file
    """
    if filename is None:
        filename = "results.csv"

    if new_file:
        create_csv_file(filename)

    with open(filename, 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in results.items():
            all_coords = []
            for keypoint, coords in value.items():
                all_coords.append([coord.flatten().tolist()[0] for coord in coords])
            writer.writerow([key, *all_coords, ';'])


def read_cfg_file(cfg_file):
    with open(cfg_file) as info:
        info_dict = yaml.load(info, Loader=yaml.FullLoader)
    return info_dict


def prepare_images(folder):
    import os
    import cv2.cv2 as cv2
    from dt_predictor import DtPredictor
    from visualization import visualize_keypoints
    list_images = os.listdir(folder)
    predictor = DtPredictor(device="cpu")

    for im_name in list_images:
        im = cv2.imread(os.path.join(folder, im_name))
        h, w, _ = im.shape

        # step 0
        im_to_analise = im.copy()[:, :w // 2, :]
        background = im.copy()
        kps = predictor.get_kps(im_to_analise, instance=0)
        visualize_keypoints(kps, im_to_analise, threshold=0., scale=.9)
        background[:, :w // 2, :] = im_to_analise
        cv2.imwrite(os.path.join(folder, "step0"+im_name), background)

        # step 1
        im_to_analise = im.copy()[:, w // 2:, :]
        background = im.copy()
        kps = predictor.get_kps(im_to_analise, instance=0)
        visualize_keypoints(kps, im_to_analise, threshold=0., scale=.9)
        background[:, w // 2:, :] = im_to_analise
        cv2.imwrite(os.path.join(folder, "step1"+im_name), background)

    print(list_images)


def example(detectron=False, mediapipe=False, mode=None):
    """Shows example of use"""
    # test file
    import cv2.cv2 as cv2
    from visualization import visualize_keypoints
    im = cv2.imread("images/test.jpg")

    # example of mediapipe inference
    if mediapipe:
        from mp_predictor import MpipePredictor
        predictor = MpipePredictor(detection_thr=.8, tracking_thr=.9)
        kps = predictor.get_keypoints(im)
        visualize_keypoints(kps, im, threshold=0., mode=mode, scale=.9)
        cv2.imshow("", im)
        cv2.waitKey(0)

    # example of detectron2 inference on cpu
    if detectron:
        from dt_predictor import DtPredictor
        predictor = DtPredictor(device="cpu")
        kps = predictor.get_kps(im, instance=0)
        visualize_keypoints(kps, im, threshold=0., mode=mode, scale=.9)
        cv2.imshow("", im)
        cv2.waitKey(0)


if __name__ == "__main__":
    pass
