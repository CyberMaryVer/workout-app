# -*- coding: utf-8 -*-
# Detectron2 & Mediapipe implementation
###########################
try:
    import cv2.cv2 as cv2
except Exception as e:
    import cv2
###########################
import mediapipe as mp
import os
from visualization import CvFpsCalc, visualize_keypoints
from geometry import get_central_points

_KEYPOINT_THRESHOLD = .5


def get_updated_keypoints(keypoint_dict):
    points_to_connect = get_central_points(keypoint_dict)
    keypoint_dict.update(points_to_connect)
    return keypoint_dict


class MpipePredictor(mp.solutions.pose.Pose):
    def __init__(self, detection_thr, tracking_thr=.99, path_to_video=None, static=False, instance=0):
        super().__init__(static_image_mode=static, min_detection_confidence=detection_thr,
                         min_tracking_confidence=tracking_thr)
        self.instance = instance
        self.path_to_video = path_to_video
        if self.path_to_video is not None:
            self.video = cv2.VideoCapture(self.path_to_video)
            self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.scale = max(self.height, self.width) / 850
            self.frames_per_second = self.video.get(cv2.CAP_PROP_FPS)
            self.num_frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.basename = os.path.basename(self.path_to_video)
        self.keypoints = {}
        self.tracking = {}

    def run_on_webcam(self, skeleton=1, mode=None, joints=True, threshold=.5, side=None, draw_invisible=False,
                      save_output=False, save_path=None, color_mode=None):
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cvFpsCalc = CvFpsCalc(buffer_len=10)
        scale = cam.get(3) / 850
        output_file = None
        cam_width, cam_height = int(cam.get(3)), int(cam.get(4))
        # print(cam_width, cam_height)
        with self:
            if save_output:
                if save_path is None:
                    output_filename = "webcam_out.mp4"
                else:
                    output_filename = save_path

                output_file = cv2.VideoWriter(
                    filename=output_filename,
                    fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                    fps=20.,
                    frameSize=(cam_width, cam_height),
                    isColor=True, )

            while cv2.waitKey(1) != 27:

                ret, frame = cam.read()
                frame = cv2.flip(frame, 1)

                try:
                    outputs = self.get_keypoints(frame)
                    kps = get_updated_keypoints(outputs)
                    frame = visualize_keypoints(kps, frame, skeleton=skeleton, dict_is_updated=True, joints=joints,
                                                threshold=threshold, side=side, mode=mode, scale=scale * 2)

                except Exception as e:
                    # print(e)
                    pass

                display_fps = cvFpsCalc.get()
                frame = cv2.putText(frame, "FPS:" + str(display_fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                    1.0, (0, 255, 0), 2, cv2.LINE_AA)
                if save_output:
                    output_file.write(frame)

                cv2.imshow('frame', frame)

        cam.release()
        if save_output:
            output_file.release()
        else:
            cv2.destroyAllWindows()

    def run_on_video(self, side=None, skeleton=1, mode=None, joints=True, threshold=.5, color_mode=None,
                     draw_invisible=False, save_output=False, save_path=None, debug_params=None):
        if self.path_to_video is None:
            return

        output_file = None

        with self:
            if save_output:
                if save_path is None:
                    output_filename = os.path.join(self.basename.split('.')[0] + "_out.mp4")
                else:
                    output_filename = save_path

                output_file = cv2.VideoWriter(
                    filename=output_filename,
                    fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                    fps=float(self.frames_per_second),
                    frameSize=(self.width, self.height),
                    isColor=True, )

            frame_gen = self._frame_from_video(self.video)
            for frame in frame_gen:
                keypoints = self.get_keypoints(frame)
                frame = visualize_keypoints(keypoints, frame, skeleton=skeleton, side=side, mode=mode,
                                            scale=self.scale, threshold=threshold, color_mode=color_mode,
                                            draw_invisible=draw_invisible, joints=joints, dict_is_updated=False)
                if save_output:
                    output_file.write(frame)

                yield frame, keypoints

            self.video.release()

            if save_output:
                output_file.release()
            else:
                cv2.destroyAllWindows()

    def get_keypoints(self, img, get3d=False):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, _ = img.shape
        results = self.process(img)
        if results is None:
            return
        idxs = mp.solutions.pose.PoseLandmark
        try:
            self.keypoints = {'nose': results.pose_landmarks.landmark[idxs.NOSE],
                              'left_eye': results.pose_landmarks.landmark[idxs.LEFT_EYE],
                              'right_eye': results.pose_landmarks.landmark[idxs.RIGHT_EYE],
                              'left_ear': results.pose_landmarks.landmark[idxs.LEFT_EAR],
                              'right_ear': results.pose_landmarks.landmark[idxs.RIGHT_EAR],
                              'left_shoulder': results.pose_landmarks.landmark[idxs.LEFT_SHOULDER],
                              'right_shoulder': results.pose_landmarks.landmark[idxs.RIGHT_SHOULDER],
                              'left_elbow': results.pose_landmarks.landmark[idxs.LEFT_ELBOW],
                              'right_elbow': results.pose_landmarks.landmark[idxs.RIGHT_ELBOW],
                              'left_wrist': results.pose_landmarks.landmark[idxs.LEFT_WRIST],
                              'right_wrist': results.pose_landmarks.landmark[idxs.RIGHT_WRIST],
                              'left_hip': results.pose_landmarks.landmark[idxs.LEFT_HIP],
                              'right_hip': results.pose_landmarks.landmark[idxs.RIGHT_HIP],
                              'left_knee': results.pose_landmarks.landmark[idxs.LEFT_KNEE],
                              'right_knee': results.pose_landmarks.landmark[idxs.RIGHT_KNEE],
                              'left_ankle': results.pose_landmarks.landmark[idxs.LEFT_ANKLE],
                              'right_ankle': results.pose_landmarks.landmark[idxs.RIGHT_ANKLE]}
            keypoints_dict = {}
            for name, obj in self.keypoints.items():
                keypoints_dict.update({name: self._get_coords(obj, img.shape, get3d)})

        except AttributeError:
            print("#Error: no object found, pose is not detected.")
            return

        return keypoints_dict

    def _get_coords(self, keypoint, img_shape=None, with_z=True):
        """get coords for keypoint"""
        x, y, z, vis = keypoint.x, keypoint.y, keypoint.z, keypoint.visibility

        if img_shape is not None:
            image_height, image_width, _ = img_shape
            image_depth = (image_height + image_width) / 2
            x, y, z = int(x * image_width), int(y * image_height), int(z * image_depth)

        if not with_z:
            return x, y, vis

        return x, y, z, vis

    def _frame_from_video(self, video):
        # while video.isOpened():
        f = 0
        while f < self.num_frames:
            success, frame = video.read()
            if success:
                yield frame
                f += 1
            else:
                break


if __name__ == "__main__":
    from utils import example
    example(mediapipe=True, mode="angles")
