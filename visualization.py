# -*- coding: utf-8 -*-
# Detectron2 & Mediapipe implementation

###########################
try:
    import cv2.cv2 as cv2
except Exception as e:
    import cv2
###########################
import textwrap
import numpy as np
from collections import deque
from geometry import get_central_points, define_symmetry, center_of_gravity, get_angle_dict

_KEYPOINT_THRESHOLD = .05

CONNECTION_RULES = [
    ('left_ear', 'left_eye', (255, 255, 255)),  # 0
    ('right_ear', 'right_eye', (255, 255, 255)),  # 1
    ('left_eye', 'nose', (255, 255, 255)),  # 2
    ('nose', 'right_eye', (255, 255, 255)),  # 3
    ('left_shoulder', 'right_shoulder', (0, 255, 55)),  # 4
    ('left_shoulder', 'left_elbow', (0, 255, 0)),  # 5
    ('right_shoulder', 'right_elbow', (0, 255, 0)),  # 6
    ('left_elbow', 'left_wrist', (0, 255, 0)),  # 7
    ('right_elbow', 'right_wrist', (0, 255, 0)),  # 8
    ('left_hip', 'right_hip', (0, 255, 55)),  # 9
    ('left_hip', 'left_knee', (0, 255, 0)),  # 10
    ('right_hip', 'right_knee', (0, 255, 0)),  # 11
    ('left_knee', 'left_ankle', (0, 255, 0)),  # 12
    ('right_knee', 'right_ankle', (0, 255, 0))  # 13
]
CONNECTION_RULES_ADDITION1 = [
    ('neck_center', 'hip_center', (0, 255, 0)),
    ('neck_center', 'nose', (0, 255, 0)),
]

CONNECTION_RULES_ADDITION2 = [
    ('head_center', 'left_shoulder', (255, 255, 255)),
    ('head_center', 'right_shoulder', (255, 255, 255)),
    ('head_center', 'neck_center', (255, 255, 255))
]

CONNECTION_RULES_ADDITION3 = [
    ('hip_center', 'left_hip', (0, 255, 0)),
    ('hip_center', 'right_hip', (0, 255, 0)),
    ('neck_center', 'left_shoulder', (0, 255, 0)),
    ('neck_center', 'right_shoulder', (0, 255, 0)),
]


def generate_img(box_shape, color):
    """Creates colored rectangle"""
    img = np.full(box_shape, color)
    img = np.array(img, dtype=np.uint8)
    return img


def insert_image(main_img, insert_img, y=5, x=5):
    """Inserts image in the right upper corner"""
    h, w, _ = main_img.shape
    h_, w_, _ = insert_img.shape
    if h_ + y > h or w_ + x > w:
        print("Warning: the image to be inserted does not match the main image")
        return main_img
    main_img[y:insert_img.shape[0] + y, w - insert_img.shape[1] - x:w - x] = insert_img
    return main_img


def draw_box_with_multiline_text(box_shape: tuple, text: str, border: int = 5, color=(255, 255, 255), font_scale=1.,
                                 font_thickness=2, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, auto=True):
    """
    This function draws multi-line text in a box with given shape
    """
    img = generate_img(box_shape, color)
    text_color = [255 - i for i in color]
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)
    w_text, h_text = text_size[0]
    gap = text_size[1]
    one_char = w_text / len(text)
    chars_in_line = int((box_shape[1] - 2 * border) / one_char)

    x = border
    y = border + h_text

    if auto:
        wrapped = textwrap.wrap(text, width=chars_in_line)
    else:
        wrapped = text.split('\n')

    for line in wrapped:
        cv2.putText(img, line, (x, y), font, font_scale, text_color,
                    font_thickness, lineType=cv2.LINE_AA)
        y = y + h_text + gap

    return img


def draw_text(img, text, font=cv2.FONT_HERSHEY_COMPLEX_SMALL, position=(10, 10), font_scale=0.6,
              font_thickness=1, text_color=(0, 0, 0), text_color_bg=(255, 255, 255), alignment="center"):
    """
    This function draws text in a coloured rectangle
    """

    x, y = [int(x) for x in position]

    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    # space
    text_w, text_h = [int(x) for x in text_size]
    cv2.rectangle(img, position, (x + text_w, y + text_h + 5), text_color_bg, -1, lineType=cv2.LINE_AA)
    cv2.putText(img, text, (x, y + text_h), font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)

    return img


def draw_angle_in_circle(img: object, angle: int, xy_coordinates: tuple, scale=None, symmetry=True, antialiasing=True):
    """
    This function draws angle value in a circle
    """
    if antialiasing:
        linetype = cv2.LINE_AA
    else:
        linetype = cv2.LINE_4

    scale = 1 if scale is None else scale
    thickness = 1 if scale is None else int(scale * 1)
    shift = 0 if scale is None else int((len(str(angle)) == 3) * scale * 2.4) + \
                                    int((len(str(angle)) == 1) * scale * -4.4)
    radius = 12 if scale is None else int(scale * 11)
    font_scale = .3 if scale is None else .3 * scale
    circle_color = (255, 255, 255)
    text_color = (0, 0, 0)

    if symmetry:
        radius = 16 if scale is None else int(scale * 15)
        font_scale = .3 if scale is None else .3 * scale
        circle_color = (0, 200, 200)

    x, y = xy_coordinates
    x1, y1 = x - int(8 * scale) - shift, y - int(6 * scale)
    x2, y2 = x - int(8 * scale) - shift, y + int(4 * scale)

    img = cv2.circle(img, (x, y), radius=radius, color=circle_color, thickness=-1, lineType=linetype)
    img = cv2.putText(img, f"{angle}", (x2, y2), cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale,
                      color=text_color, thickness=thickness, lineType=linetype)
    if symmetry:
        img = cv2.putText(img, f"+/-", (x1, y1), cv2.FONT_HERSHEY_COMPLEX, fontScale=font_scale * .7,
                          color=(25, 100, 25), thickness=1, lineType=linetype)

    return img


def draw_box_with_text(img, text=None, edge_color=(255, 255, 255), border=2, multiline=True, font_scale=.8,
                       font_thickness=2, font_color=(0, 0, 0)):
    """
    This function draws a box around the image with or w/o text
    """
    img = cv2.copyMakeBorder(img, 10 * border, border, border, border, cv2.BORDER_CONSTANT, value=edge_color)

    if text is not None:
        if multiline:
            box_w = int(img.shape[1] * .6)
            box_h = border * 8
            box_s = (box_h, box_w, 3)
            textbox = draw_box_with_multiline_text(box_shape=box_s, text=text, color=edge_color, font_scale=font_scale,
                                                   font_thickness=font_thickness, font=cv2.FONT_HERSHEY_SIMPLEX, )
            img = insert_image(img, textbox, 10, 15)  # right alignment

        else:
            x = y = border
            img = cv2.putText(img, text, (x + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color,
                              font_thickness, lineType=cv2.LINE_AA)

    return img


def get_updated_keypoint_dict(keypoint_dict):
    points_to_connect = get_central_points(keypoint_dict)
    keypoint_dict.update(points_to_connect)
    return keypoint_dict


def draw_skeleton(img, keypoints, side=None, threshold=0., thickness=1, headless=False, draw_invisible=False,
                  color_invisible=(188, 188, 188), antialiasing=True):
    """
    This function draws connections on the image and returns image
    """
    rules = CONNECTION_RULES
    additional_rules = CONNECTION_RULES_ADDITION1
    if not headless:
        for addition in additional_rules:
            rules.append(addition)

    if antialiasing:
        linetype = cv2.LINE_AA
    else:
        linetype = cv2.LINE_4

    if len(keypoints) == 17:
        keypoints = get_updated_keypoint_dict(keypoints)

    if headless:
        xrs, yrs, _ = keypoints["right_shoulder"]
        xls, yls, _ = keypoints["left_shoulder"]
        x1, y1, t1 = keypoints["nose"]
        x2, y2, t2 = keypoints["neck_center"]
        # calculate point to connect the lines
        x = (x1 + x2) // 2
        y = (y1 + y2) // 2
        t = (t1 + t2) / 2
        keypoints.update({"head_center": (x, y, t)})
        rules.append(additional_rules[0])
        additional_rules = CONNECTION_RULES_ADDITION2
        rules = rules[4:-1]
        for addition in additional_rules:
            rules.append(addition)

    if side is not None:
        additional_rules = CONNECTION_RULES_ADDITION3
        for addition in additional_rules:
            rules.append(addition)

        if side == "L":
            side, other = "left", "right"
        else:
            side, other = "right", "left"

    for rule in rules:
        # set visibility
        draw_connection = True

        if side is not None:
            # print(side, rule[0], rule[1], side in rule[0] and side in rule[1])
            if side in rule[0] and side in rule[1]:
                draw_connection = True
            elif "center" in rule[0]:
                draw_connection = True
                if other in rule[1]:
                    draw_connection = False
            else:
                draw_connection = False

        if draw_connection:
            try:
                p1, p2, color = rule
                x1, y1, t1 = keypoints[p1]
                x2, y2, t2 = keypoints[p2]

                if t1 > threshold and t2 > threshold:
                    img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color,
                                   thickness=thickness, lineType=linetype)
                elif draw_invisible:
                    color = color_invisible
                    img = cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color,
                                   thickness=thickness, lineType=linetype)
                else:
                    pass
            except Exception as e:
                print("Draw connection error:", e)
                pass

    return img


def draw_joints(img, keypoints, threshold=0., side=None, headless=False, color=(255, 255, 255), draw_invisible=False,
                color_invisible=(188, 188, 188), point_radius=6, antialiasing=True):
    """
    This function draws keypoints on the image and returns image
    """
    if antialiasing:
        linetype = cv2.LINE_AA
    else:
        linetype = cv2.LINE_4

    if len(keypoints) == 17:
        keypoints = get_updated_keypoint_dict(keypoints)

    if side is not None:
        side = "right" if side == "R" else "left"

    keypoints_visibility = {x: True for x in keypoints.keys()}

    for name, i in keypoints.items():
        # set visibility for each side
        draw_point = True

        if side is not None and side not in name:
            draw_point = False
            keypoints_visibility.update({name: draw_point})

        if headless:
            rules_for_head = ["eye", "ear", "nose"]

            if not name == "nose":
                name = name.split("_")[1]

            if name in rules_for_head:
                draw_point = False
                keypoints_visibility.update({name: draw_point})

        if draw_point:
            x, y, t = int(i[0]), int(i[1]), i[2]
            if t > threshold:
                img = cv2.circle(img, (x, y), radius=point_radius, color=color, thickness=-1, lineType=linetype)
            elif t < threshold and draw_invisible:
                color = color_invisible
                img = cv2.circle(img, (x, y), radius=point_radius, color=color, thickness=-1, lineType=linetype)
            else:
                pass

    return img, keypoints_visibility


def visualize_keypoints(keypoint_dict, im_wk, skeleton=1, side=None, mode=None, scale=None, threshold=None,
                        color_mode=None, draw_invisible=False, joints=True, dict_is_updated=False,
                        antialiasing=True, alpha=None):
    if keypoint_dict is None:
        return im_wk
    if side not in ["L", "R", None]:
        print("Error: wrong side parameter")
        return 1

    if alpha is not None:
        im_wk_overlay = im_wk.copy()
    else:
        im_wk_overlay = None

    if antialiasing:
        linetype = cv2.LINE_AA
    else:
        linetype = cv2.LINE_4

    if not dict_is_updated:  # for compability with mediapipe predictor
        all_keypoints = get_updated_keypoint_dict(keypoint_dict)
    else:
        all_keypoints = keypoint_dict

    threshold = _KEYPOINT_THRESHOLD if threshold is None else threshold

    if color_mode is None:
        point_color1 = point_color2 = (255, 255, 255)
    else:
        point_color1 = (0, 255, 0)

    if scale is not None:
        thickness = int(scale * 2.4)
        point_radius = int(scale * 6.4)
        font_scale = .3 * scale
    else:
        thickness = 2
        point_radius = 6
        font_scale = .3

    if skeleton != 0:
        headless = True if skeleton == 2 else False
        im_wk = draw_skeleton(im_wk, all_keypoints, side=side, threshold=threshold, thickness=thickness,
                              headless=headless, draw_invisible=draw_invisible, antialiasing=antialiasing)

    if joints:
        headless = True if skeleton == 2 else False
        im_wk, vis = draw_joints(im_wk, all_keypoints, threshold=threshold, side=side, headless=headless,
                                 color=point_color1, draw_invisible=draw_invisible, point_radius=point_radius,
                                 antialiasing=antialiasing)

        for name, visibility in vis.items():

            if mode == "keypoints_names" and visibility:
                # """displays keypoint names"""
                x, y, _ = all_keypoints[name]
                name = name.replace("_", " ")
                print(x, y)
                im_wk = draw_text(im_wk, name, font=cv2.FONT_HERSHEY_SIMPLEX, position=(int(x), int(y)),
                                  font_scale=font_scale)

            elif mode == "angles" and visibility:
                # """displays angles values"""
                angles_dict = get_angle_dict(keypoint_dict, side=side, dict_is_updated=True)

                for kp, kp_data in angles_dict.items():
                    x, y = int(kp_data[1][0]), int(kp_data[1][1])
                    angle_value = kp_data[0]
                    im_wk = draw_angle_in_circle(im_wk, angle_value, (x, y), scale=scale, symmetry=False,
                                                 antialiasing=antialiasing)

            elif mode == "symmetry" and visibility:
                # """displays symmetry breaks"""
                angles_dict = get_angle_dict(keypoint_dict, dict_is_updated=True)
                sym = define_symmetry(angles_dict)

                for key in sym.keys():
                    if "hip_center" in key:
                        key1 = key2 = "hip_center"
                    elif "neck_center" in key:
                        key1 = key2 = "neck_center"
                    else:
                        key1 = "right" + key
                        key2 = "left" + key
                    # ±, º
                    x1, y1, _ = all_keypoints[key1]
                    x2, y2, _ = all_keypoints[key2]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    deviation = int(abs(sym[key]))
                    im_wk = draw_angle_in_circle(im_wk, deviation, (x1, y1), scale=scale, antialiasing=antialiasing)
                    im_wk = draw_angle_in_circle(im_wk, deviation, (x2, y2), scale=scale, antialiasing=antialiasing)

                # if len(sym) > 0:
                    # print(sym)

            elif mode == "gravity_center":
                # """displays center of gravity"""
                xline, yline = center_of_gravity(all_keypoints)
                x1, y1 = [int(p) for p in xline]
                x2, y2 = [int(p) for p in yline]
                div = 1.0 * (x2 - x1) if x2 != x1 else .00001
                a = (1.0 * (y2 - y1)) / div
                b = -a * x1 + y1
                y1_, y2_ = 0, im_wk.shape[1]
                x1_ = int((y1_ - b) / a)
                x2_ = int((y2_ - b) / a)
                line_color = (255, 100, 100)
                point_color = (255, 100, 100)
                line_length = max(im_wk.shape) // 3

                # draw lines
                im_wk = cv2.line(im_wk, (x1_, y1_), (x2_, y2_), color=line_color, thickness=1, lineType=linetype)
                im_wk = cv2.line(im_wk, (x2 - line_length, y1), (x1 + line_length, y1), color=line_color, thickness=1,
                                 lineType=linetype)
                # draw points
                im_wk = cv2.circle(im_wk, (x1, y1), radius=point_radius * 2, color=point_color, thickness=-1,
                                   lineType=linetype)
                im_wk = cv2.circle(im_wk, (x2, y2), radius=point_radius, color=(255, 255, 255), thickness=-1,
                                   lineType=linetype)

    if alpha is not None:
        im_wk = cv2.addWeighted(im_wk_overlay, alpha, im_wk, 1 - alpha, 1)

    return im_wk


def draw_points_by_name(img, keypoints, names: list, radius=6, color=(0, 0, 255)):
    for error in names:
        x, y, _ = keypoints[error]
        img = cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)

    return img


class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv2.getTickCount()
        self._freq = 1000.0 / cv2.getTickFrequency()
        # deque = collections.deque
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv2.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded


if __name__ == "__main__":
    from utils import example

    example(mediapipe=True, mode="angles")
