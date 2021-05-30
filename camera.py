import cv2.cv2 as cv2
import threading
import time
from datetime import datetime
import logging
from utils import example
from mp_workout import MpipeWorkout
from mp_workout import DEFAULT_MODE, DEFAULT_SKELETON, DEFAULT_BORDER
from visualization import draw_box_with_text, get_updated_keypoint_dict, draw_points_by_name, \
    draw_angle_in_circle, visualize_keypoints, draw_box_with_multiline_text, insert_image
from geometry import *
from workout_configs.catalog import WorkoutDataLoader
from mp_predictor import MpipePredictor

logger = logging.getLogger(__name__)

thread = None


class Camera:
    def __init__(self, fps=20, video_source=0):
        logger.info(f"Initializing camera class with {fps} fps and video_source={video_source}")
        self.fps = fps
        self.video_source = video_source
        self.camera = cv2.VideoCapture(self.video_source)
        self.history_cash = 5  # We want a max of 5s history to be stored, thats 5s*fps
        self.max_frames = self.history_cash * self.fps
        self.frames = []
        self.isrunning = False

        self.config = WorkoutDataLoader.dumbbell_lateral_raise()
        self.predictor = MpipePredictor(detection_thr=.8, tracking_thr=.9)
        self.WINDOW_NAME = self.config.WINDOW_NAME
        self.ANGLES = self.config.ANGLES
        self.HIGH_LOW = self.config.HIGH_LOW
        self.ERRORS = self.config.ERRORS
        self.DESCRIPTION = self.config.DESCRIPTION
        self.MET = self.config.MET
        self.IMAGE = self.config.IMAGE_PATH
        self.MODE = "angles"
        self.SKELETON = DEFAULT_SKELETON
        self.BORDER = DEFAULT_BORDER
        self.W = None
        self.H = None
        self.START_TIME = 0
        self.STEPS = 6
        self.WEIGHT = 60
        self.WRONG = 0
        self.CORRECT = 0
        self.delay = 0
        self.current = 0
        self.state = 0
        self.reps = 0
        self.txt = "DUMBBELL LATERAL RAISE"
        self.m = MpipeWorkout(weight=self.WEIGHT, workout_reps=self.STEPS, config=self.config)

    def run(self):
        logging.debug("Preparing thread")
        global thread
        if thread is None:
            logging.debug("Creating thread")
            thread = threading.Thread(target=self._capture_loop, daemon=True)
            logger.debug("Starting thread")
            self.isrunning = True
            thread.start()
            logger.info("Thread started")

    def _capture_loop(self):
        dt = 1 / self.fps
        logger.debug("Observation started")
        while self.isrunning:
            ret, frame = self.camera.read()
            if ret:
                if len(self.frames) == self.max_frames:
                    self.frames = self.frames[1:]
                self.frames.append(frame)
            time.sleep(dt)
        logger.info("Thread stopped successfully")

    def stop(self):
        logger.debug("Stopping thread")
        self.isrunning = False

    def get_frame(self, _bytes=True):
        if len(self.frames) > 0:
            if _bytes:
                img = cv2.imencode('.png', self.frames[-1])[1].tobytes()
            else:
                img = self.frames[-1]
        else:
            with open("images/not_found.jpeg", "rb") as f:
                img = f.read()
        return img

    def get_pose(self, _bytes=True, _test=False, _workout=True, insert_img=None):
        ret = True
        if len(self.frames) > 0:
            img = self.frames[-1]
            img = cv2.flip(img, 1)

            if _test:
                img = example(mediapipe=True, show_result=False, im=img)
                img = draw_box_with_text(img, "TEST", border=5)

            elif _workout:
                self.H, self.W = img.shape[:2]
                self.START_TIME = time.time()
                self.current += 1

                if not self.reps + 1 == self.STEPS:

                    try:
                        outputs = self.predictor.get_keypoints(img)
                        kps = get_updated_keypoint_dict(outputs)
                        angles = get_angle_dict(kps, dict_is_updated=True)

                        img, state_, self.txt, self.reps = self._check_exercise(img=img, kps=kps, state=self.state,
                                                                                angles=angles, reps=self.reps,
                                                                                scale=1)

                        if state_ == -1:
                            if self.delay < 10:  # ################################# #
                                print("HOLD!!!!")
                                self.delay += 1
                            else:
                                self.delay = 0

                        if state_ > self.state and self.reps != self.STEPS:
                            print("YOU GOT IT!")

                        elif state_ == -1 and self.reps == self.STEPS:
                            print("NICE!" * 1000)

                        self.state = state_

                    except Exception as e:
                        # print(e)
                        self.txt = "Try to stay visible for the camera"
                        img = draw_box_with_text(img, self.txt, edge_color=(255, 255, 255), border=6)

                    print(f"reps = {self.reps}, steps = {self.STEPS}")
                    img = insert_image(img, insert_img, x=440, y=35)

                else:
                    print("WELL DONE!")
                    img = self._draw_final(img, (100, 255, 100))
                    ret = False

            if _bytes:
                img = cv2.imencode('.png', img)[1].tobytes()

        else:
            with open("images/not_found.jpeg", "rb") as f:
                img = f.read()

        return img, ret

    def _check_exercise(self, img, kps, state=0, reps=0, angles=None, scale=1.):
        """Checks if exercise is done correctly"""
        edge_color = (255, 255, 255)
        paste_check = False  # draws a check on the top of the frame

        state = 0 if state == -1 else state
        condition1 = self.ANGLES
        condition2 = self.HIGH_LOW
        condition3 = self.ERRORS
        description = self.DESCRIPTION
        angle_condition = condition1[state]
        high_low_condition = condition2[state]
        error_condition = condition3[state]
        txt = description[state]  # first line is for the start only
        steps = len(condition1)

        # check conditions - state condition
        state_check = self.m.compare_angles(angles, angle_condition)

        if not self.m.compare_high_low(kps, high_low_condition):
            state_check.append(high_low_condition)

        if len(state_check) == 0:
            if state != steps - 1:
                state += 1
                # txt = "CORRECT POSITION"

            elif state == steps - 1:
                state = -1
                reps += 1
                txt = f"GOOD WORK!"
                # edge_color = (0, 255, 0)
                # paste_check = True

            if state == -1 and reps == self.STEPS:
                img_final = self.m._draw_final(img, reps)
                date_and_time = datetime.now().strftime("%d%m%Y%H%M")
                img_name = f"{self.WINDOW_NAME}_{date_and_time}.jpg"
                img_path = "gallery/" + img_name
                self.FINAL = img_final
                cv2.imwrite(img_path, img_final)

        else:
            txt_debug = f"WAITING FOR THE NEXT POSITION. {txt.upper()}"
            print(txt_debug)

        # check conditions - correct position
        visual_check = self.m.compare_angles(angles, error_condition)

        if len(visual_check) == 0:
            self.CORRECT += 1
        else:
            self.WRONG += 1

        img = draw_points_by_name(img, kps, visual_check, 20)
        img = visualize_keypoints(kps, img, skeleton=self.SKELETON, dict_is_updated=True, threshold=.7,
                                  scale=scale * 2, mode=self.MODE, alpha=.6)
        img = draw_angle_in_circle(img, reps, (self.W - 60, 70), scale=4, symmetry=False)
        img = draw_box_with_text(img, txt.upper(), edge_color=edge_color, border=self.BORDER, font_scale=.7)

        return img, state, txt, reps

    def _draw_final(self, img, color=(200, 255, 200)):
        """Draws and saves final frame with result"""

        total_frames = self.WRONG + self.CORRECT
        txt_for_frame = f"Exercise {self.WINDOW_NAME} is finished" \
                        f"\nYou have done {self.reps} repetitions" \
                        f"\nAccuracy: {100 * self.CORRECT // total_frames}%"

        img = draw_box_with_text(img, text="WELL DONE!!!", edge_color=color, border=self.BORDER)
        textbox_shape = (self.H // 5, self.W - 5, 3)
        textbox = draw_box_with_multiline_text(box_shape=textbox_shape, text=txt_for_frame, color=color, auto=False,
                                               font_scale=.6, font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)
        img = insert_image(main_img=img, insert_img=textbox, y=55)

        return img

    def restart(self):
        self.WRONG = 0
        self.CORRECT = 0
        self.delay = 0
        self.current = 0
        self.state = 0
        self.reps = 0
        self.txt = "DUMBBELL LATERAL RAISE"
