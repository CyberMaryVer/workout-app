from visualization import *
from geometry import *
from mp_predictor import MpipePredictor
from workout_configs.catalog import WorkoutDataLoader
from time import time
from datetime import datetime
import argparse

MODES = ["angles", "symmetry", "gravity_center"]
SKELETONS = [0, 1, 2]
DEFAULT_CHECK = cv2.imread("images/check.jpg")
DEFAULT_SCALE = 1
DEFAULT_SAVE_PATH = "result.mp4"
DEFAULT_SAVE_OUTPUT = False
DEFAULT_BORDER = 18
DEFAULT_MODE = None
DEFAULT_SKELETON = 2


# TODO: replace code with draw_frame method
# TODO: implement get_frame method
# TODO: web API

class MpipeWorkout:
    def __init__(self, weight, workout_reps, config: WorkoutDataLoader):
        self.STEPS = workout_reps
        self.WEIGHT = weight
        self.WINDOW_NAME = config.WINDOW_NAME
        self.ANGLES = config.ANGLES
        self.HIGH_LOW = config.HIGH_LOW
        self.ERRORS = config.ERRORS
        self.DESCRIPTION = config.DESCRIPTION
        self.MET = config.MET
        self.IMAGE = config.IMAGE_PATH
        self.SCALE = DEFAULT_SCALE
        self.MODE = DEFAULT_MODE
        self.SKELETON = DEFAULT_SKELETON
        self.CHECK = DEFAULT_CHECK
        self.BORDER = DEFAULT_BORDER
        self.SAVE_PATH = DEFAULT_SAVE_PATH
        self.SAVE_OUTPUT = DEFAULT_SAVE_OUTPUT
        self.PATH_TO_VIDEO = None
        self.OUTPUT_FILE = None
        self.FINAL = None
        self.SHOW_FINAL = True
        self.START_TIME = None
        self.W = None
        self.H = None
        self.WRONG = 0
        self.CORRECT = 0
        self.WORKOUT_IMAGES = []

    def _calculate_calories(self, duration):
        """Calculator of calories burned
            ............................
            MET ESTIMATION:
            Calisthenics (e.g. pushups, sit-ups, pullups, jumping jacks), heavy, vigorous effort – 8.0
            Circuit training, including some aerobic movement with minimal rest, general – 8.0
            Weightlifting, powerlifting or bodybuilding, vigorous effort – 6.0
            Stair-treadmill ergometer, general – 9.0
            Mild stretching – 2.5
            Jog/walk combination (jogging component of less than 10 minutes) – 6.0
            Jogging, in place or 5 mph (8 km/h) – 8.0
        """
        duration = duration / 60
        your_weight = 60
        total_calories = duration * (self.MET * your_weight * 3.5) / 200

        return total_calories

    def _draw_frame(self, img, reps, kps, txt, scale, edge_color=(0, 255, 0), paste_check=True):
        img = visualize_keypoints(kps, img, skeleton=self.SKELETON, dict_is_updated=True, threshold=.7,
                                  scale=scale, mode=self.MODE)
        img = draw_angle_in_circle(img, reps, (self.W - 60, 70), scale=4, symmetry=False)
        img = draw_box_with_text(img, txt.upper(), edge_color=edge_color, border=self.BORDER, multiline=True)

        if paste_check and self.CHECK is not None:
            img = insert_image(img, self.CHECK, 10, 10)
        return img

    def _draw_final(self, img, reps, improve=False, color=(200, 255, 200)):
        """Draws and saves final frame with result"""
        total_time = time() - self.START_TIME
        total_calories = self._calculate_calories(total_time)
        total_frames = self.WRONG + self.CORRECT
        txt_for_frame = f"Exercise {self.WINDOW_NAME} is finished" \
                        f"\nYou have done {reps} repetitions for {total_time:.2f} sec" \
                        f"\nYou burned {total_calories:.2f} calories" \
                        f"\nAccuracy: {100 * self.CORRECT // total_frames}%."

        if improve:
            img = self.improve_photo(img)

        img = draw_box_with_text(img, text="WELL DONE!!!", edge_color=color, border=self.BORDER)
        textbox_shape = (self.H // 5, self.W - 5, 3)
        textbox = draw_box_with_multiline_text(box_shape=textbox_shape, text=txt_for_frame, color=color, auto=False,
                                               font_scale=.6, font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)
        img = insert_image(main_img=img, insert_img=textbox, y=55)

        return img

    def _check_exercise(self, img, kps, state=0, reps=0, angles=None, scale=1., verbose=True):
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
        state_check = self.compare_angles(angles, angle_condition)

        if not self.compare_high_low(kps, high_low_condition):
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
                img_final = self._draw_final(img, reps)
                date_and_time = datetime.now().strftime("%d%m%Y%H%M")
                img_name = f"{self.WINDOW_NAME}_{date_and_time}.jpg"
                img_path = "gallery/" + img_name
                self.FINAL = img_final
                cv2.imwrite(img_path, img_final)

        else:
            txt_debug = f"WAITING FOR THE NEXT POSITION. {txt.upper()}"
            if verbose:
                print(txt_debug)

        # check conditions - correct position
        visual_check = self.compare_angles(angles, error_condition)

        if len(visual_check) == 0:
            self.CORRECT += 1
        else:
            self.WRONG += 1

        img = draw_points_by_name(img, kps, visual_check, 20)
        img = visualize_keypoints(kps, img, skeleton=self.SKELETON, dict_is_updated=True, threshold=.7,
                                  scale=scale, mode=self.MODE, alpha=.4)
        img = draw_angle_in_circle(img, reps, (self.W - 60, 70), scale=4, symmetry=False)
        img = draw_box_with_text(img, txt.upper(), edge_color=edge_color, border=self.BORDER, font_scale=.7)

        return img, state, txt, reps

    @staticmethod
    def test_device(source):
        """Checks if webcam exists and enabled"""
        cam = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if cam is None or not cam.isOpened():
            print(f'Warning: unable to open video source: {source}')
            return False
        cam.release()
        return True

    @staticmethod
    def compare_angles(angles, condition):
        """Finds errors by condition, returns names of incorrect points"""
        check = []
        for angle_name, limits in condition.items():
            angle = angles[angle_name][0]
            min_angle, max_angle = limits
            if angle < min_angle or angle > max_angle:
                # print(f"{angle_name}: {min_angle} < {angle} < {max_angle}")
                check.append(angle_name)
        return check

    @staticmethod
    def compare_high_low(kps, condition):
        """Checks if first point is higher"""
        p_high, p_low = condition
        if p_high is None or p_low is None:
            return True
        x1, y1, _ = kps[p_high]
        x2, y2, _ = kps[p_low]
        return y1 < y2

    @staticmethod
    def improve_photo(img):
        """Postprocessing of the frame - applies some filters"""
        img_ = img.copy()
        # img_ = cv2.detailEnhance(img_, sigma_s=20, sigma_r=0.15)
        # img_ = cv2.edgePreservingFilter(img_, flags=1, sigma_s=60, sigma_r=0.15)
        img_ = cv2.stylization(img_, sigma_s=95, sigma_r=0.95)
        img = cv2.addWeighted(img, .8, img_, .7, .5)
        return img

    @staticmethod
    def frame_from_video(num_frames, video):
        f = 0
        while f < num_frames:
            success, frame = video.read()
            if success:
                yield frame
                f += 1
            else:
                break

    def process(self):
        """Runs the stream from webcam and exercise checking process"""
        if not self.test_device(0):  # check webcam
            return

        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.W, self.H = int(cam.get(3)), int(cam.get(4))
        scale = max(self.W, self.H) / 400 * self.SCALE
        self.START_TIME = time()
        # scaled_width = int(self.W * self.SCALE)

        # open and prepare image of exercise
        for img_path in self.IMAGE:
            insert_img = cv2.imread(img_path)
            insert_img = cv2.resize(insert_img, None, None, fx=.4, fy=.4)
            self.WORKOUT_IMAGES.append(insert_img)

        delay = 0
        current = 0
        state = 0
        reps = 0
        predictor = MpipePredictor(detection_thr=.8, tracking_thr=.9)

        while cv2.waitKey(1) != 27:
            ret, frame = cam.read()

            # if frame is None:
            #     continue

            frame = cv2.flip(frame, 1)
            current += 1

            # frame = imutils.resize(frame, width=scaled_width)

            if not reps == self.STEPS:
                try:
                    outputs = predictor.get_keypoints(frame)
                    kps = get_updated_keypoint_dict(outputs)
                    angles = get_angle_dict(kps, dict_is_updated=True)

                    frame, state_, txt, reps = self._check_exercise(img=frame, kps=kps, state=state, angles=angles,
                                                                    reps=reps, scale=scale)
                    # txt_debug = f"step {state_}, txt: {txt.lower()}, reps {reps}"

                    if state_ == -1:
                        if delay < 10:  # ################################# #
                            print("HOLD!!!!")
                            delay += 1
                        else:
                            delay = 0

                    if state_ > state and reps != self.STEPS:
                        print("YOU GOT IT!")

                    elif state_ == -1 and reps == self.STEPS:
                        print("NICE!")
                        cam.release()
                        break

                    state = state_

                except Exception as e:
                    # print(e)
                    txt = "Try to stay visible for the camera"
                    frame = draw_box_with_text(frame, txt, edge_color=(255, 255, 255), border=self.BORDER)

                frame = insert_image(frame, self.WORKOUT_IMAGES[state - 1], x=440, y=35)
                cv2.imshow(self.WINDOW_NAME, frame)

                if self.SAVE_OUTPUT:
                    if self.OUTPUT_FILE is None:  # open output file when 1st frame is received
                        frame_width, frame_height, _ = [int(num) for num in frame.shape]
                        self.OUTPUT_FILE = cv2.VideoWriter(filename=self.SAVE_PATH,
                                                           fourcc=cv2.VideoWriter_fourcc(*"mp4v"), fps=40.,
                                                           frameSize=(frame_height, frame_width), isColor=True, )
                    if self.OUTPUT_FILE is not None:
                        self.OUTPUT_FILE.write(frame)

        cam.release()

        if self.SAVE_OUTPUT and self.OUTPUT_FILE is not None:
            self.OUTPUT_FILE.release()

        cv2.destroyAllWindows()

        if self.SHOW_FINAL and self.FINAL is not None:
            cv2.imshow("YOUR RESULT", self.FINAL)
            cv2.waitKey(0)

    def process_video(self):
        if self.PATH_TO_VIDEO is None:
            return

        delay = 0
        current = 0
        state = 0
        reps = 0
        predictor = MpipePredictor(detection_thr=.8, tracking_thr=.9)

        # open and prepare image of exercise TODO: class method
        for img_path in self.IMAGE:
            insert_img = cv2.imread(img_path)
            insert_img = cv2.resize(insert_img, None, None, fx=.6, fy=.4)
            self.WORKOUT_IMAGES.append(insert_img)

        # open and prepare video for processing
        video = cv2.VideoCapture(self.PATH_TO_VIDEO)
        self.START_TIME = time()
        self.W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        scale = max(self.W, self.H) / 400 * self.SCALE

        if num_frames == 0:
            print("#Error: video file is empty")
            return

        frame_gen = self.frame_from_video(num_frames, video)

        for frame in frame_gen:

            current += 1

            if not reps == self.STEPS:
                try:
                    outputs = predictor.get_keypoints(frame)
                    kps = get_updated_keypoint_dict(outputs)
                    angles = get_angle_dict(kps, dict_is_updated=True)

                    frame, state_, txt, reps = self._check_exercise(img=frame, kps=kps, state=state, angles=angles,
                                                                    reps=reps, scale=scale, verbose=False)

                    if state_ == -1:
                        if delay < 10:  # ######################################### #
                            print("HOLD!!!!")
                            delay += 1
                        else:
                            delay = 0

                    if state_ > state and reps != self.STEPS:
                        print("YOU GOT IT!")

                    elif state_ == -1 and reps == self.STEPS:
                        print("NICE!")
                        video.release()
                        break

                    state = state_

                except Exception as e:
                    # print(e)
                    txt = "Try to stay visible for the camera"
                    frame = draw_box_with_text(frame, txt, edge_color=(255, 255, 255), border=self.BORDER)

                frame = insert_image(frame, self.WORKOUT_IMAGES[state - 1], x=940, y=35) # TODO: fix coordinates

                if self.SAVE_OUTPUT:
                    if self.OUTPUT_FILE is None:  # open output file when 1st frame is received
                        print("CREATE")
                        frame_width, frame_height, _ = [int(num) for num in frame.shape]
                        self.OUTPUT_FILE = cv2.VideoWriter(filename="test.mp4",
                                                           fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                                                           fps=frames_per_second,
                                                           frameSize=(frame_height, frame_width),
                                                           isColor=True, )
                    if self.OUTPUT_FILE is not None:
                        self.OUTPUT_FILE.write(frame)

        if self.OUTPUT_FILE is not None:
            print("RELEASE")
            self.OUTPUT_FILE.release()

        if self.FINAL is not None:
            cv2.imwrite("test.jpg", self.FINAL)

    def close(self):
        self.OUTPUT_FILE = None
        self.FINAL = None
        self.SHOW_FINAL = True
        self.START_TIME = None
        self.W = None
        self.H = None

    def __enter__(self):
        """A "with" statement support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Closes all the input sources."""
        self.close()


def get_parser():
    """Gets parameters from the command line"""
    parser = argparse.ArgumentParser(description="Standard Parser")
    parser.add_argument("--test_camera", action="store_true", help="test camera")
    parser.add_argument("--test_video", action="store_true", help="process test video")
    parser.add_argument("--repetitions", type=int, default=4, help="number of repetitions")
    parser.add_argument("--weight", type=int, default=60, help="your weight")
    parser.add_argument("--config", type=str, default="dumbbell_lateral_raise", help="workout config file")
    parser.add_argument("--mode", type=str, default=None, help="visualization mode")
    parser.add_argument("--skeleton", type=str, default=None, help="skeleton mode")
    parser.add_argument("--save_video", action="store_true", help="save result as mp4 file")
    return parser


def main():
    """Main function"""
    args = get_parser().parse_args()
    test_camera = args.test_camera
    test_video = args.test_video
    reps = args.repetitions
    weight = args.weight
    workout = args.config
    mode = args.mode
    skeleton = args.skeleton
    save_video = args.save_video

    # validation
    if mode is not None and mode not in MODES:
        print("Warning: this mode was not found")
    if skeleton is not None and int(skeleton) not in SKELETONS:
        print("Error: wrong visualization parameters")
        return

    # select exercise
    try:
        method_to_call = getattr(WorkoutDataLoader, workout)
        config = method_to_call()
    except AttributeError:
        print("Selected workout not found")
        return

    # create workout with selected config
    m = MpipeWorkout(weight=weight, workout_reps=reps, config=config)

    # set visualization parameters
    if mode is not None:
        if mode not in MODES:
            mode = None
        m.MODE = mode

    if skeleton is not None:
        m.SKELETON = int(skeleton)

    if save_video:
        m.SAVE_OUTPUT = True

    if test_camera:
        print(m.test_device(1))
        print(m.test_device(0))

    elif test_video:
        path_to_video = "testdata/processed_video.mp4"
        m.PATH_TO_VIDEO = path_to_video
        m.SAVE_OUTPUT = True
        m.process_video()

    else:
        m.process()


if __name__ == "__main__":
    main()
