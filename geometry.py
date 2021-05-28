# -*- coding: utf-8 -*-
# Detectron2 & Mediapipe implementation

import numpy as np

POINTS_TO_USE = {
    'left_elbow': ['left_shoulder', 'left_elbow', 'left_wrist'],
    'right_elbow': ['right_shoulder', 'right_elbow', 'right_wrist'],
    'left_knee': ['left_hip', 'left_knee', 'left_ankle'],
    'right_knee': ['right_hip', 'right_knee', 'right_ankle'],
    'left_shoulder': ['neck_center', 'left_shoulder', 'left_elbow'],
    'right_shoulder': ['neck_center', 'right_shoulder', 'right_elbow'],
    'left_hip_center': ['neck_center', 'hip_center', 'left_hip'],
    'right_hip_center': ['neck_center', 'hip_center', 'right_hip'],
    'left_neck_center': ['hip_center', 'neck_center', 'right_shoulder'],
    'right_neck_center': ['hip_center', 'neck_center', 'right_shoulder'],
    'left_hip': ['hip_center', 'left_hip', 'left_knee'],
    'right_hip': ['hip_center', 'right_hip', 'right_knee'],
    'hip_center': ['right_ankle', 'hip_center', 'left_ankle'],
    'neck_center': ['right_shoulder', 'neck_center', 'left_shoulder']
}
VISIBLE_LEFT = ['left_elbow', 'left_knee', 'left_shoulder', 'left_hip_center', 'left_neck_center', 'left_hip']
VISIBLE_RIGHT = ['right_elbow', 'right_knee', 'right_shoulder', 'right_hip_center', 'right_neck_center',
                 'right_hip']
VISIBLE_ALL = ['left_elbow', 'left_knee', 'left_shoulder', 'left_hip', 'hip_center', 'neck_center', 'right_elbow',
               'right_knee', 'right_shoulder', 'right_hip']

CENTERS = ['hip_center', 'neck_center', 'nose']


def get_angle(p1, p2, p3):
    """
    Returns angle for three points
    """
    v1 = np.asarray(p1) - np.asarray(p2)
    v2 = np.asarray(p3) - np.asarray(p2)

    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.round(cosine_angle, 4)))
    return angle


def get_central_points(keypoint_dict):
    """
    Calculates central points
    """
    points_to_calculate_centers = [('left_hip', 'right_hip'), ('left_shoulder', 'right_shoulder')]
    points_to_connect_names = ['hip_center', 'neck_center', 'nose']
    points_to_connect = []

    for p in points_to_calculate_centers:
        x1, y1, z1 = keypoint_dict[p[0]]
        x2, y2, z2 = keypoint_dict[p[1]]
        x = float((x1 + x2) / 2)
        y = float((y1 + y2) / 2)
        z = float((z1 + z2) / 2)
        points_to_connect.append((x, y, z))
    x, y, z = keypoint_dict['nose']
    points_to_connect.append((int(x), int(y), float(z)))
    points_to_connect = {x[0]: list(x[1]) for x in zip(points_to_connect_names, points_to_connect)}

    return points_to_connect


def get_text_coords(keypoint_coords, scale):
    # (x - 18, y + 8) for 3
    # (x - 10, y + 5) for 1
    x, y = keypoint_coords
    x, y = x - int(8 * scale), y + int(4 * scale)
    return x, y


def get_angle_dict(keypoint_dict, side=None, dict_is_updated=False):
    """
    Returns dictionary with angle value for each point
    """
    keypoint_dict_ = keypoint_dict.copy()

    angles_dict = {}
    for point_name, i in POINTS_TO_USE.items():
        # set visibility for each side
        if side == "L":
            visible = VISIBLE_LEFT
        elif side == "R":
            visible = VISIBLE_RIGHT
        else:
            visible = POINTS_TO_USE.keys()  # VISIBLE_ALL

        if point_name in visible:
            a, b, c = keypoint_dict_[i[0]], keypoint_dict_[i[1]], keypoint_dict_[i[2]]
            angle = get_angle(a, b, c)
            angle = int(angle)
            text_cords = keypoint_dict_[i[1]]
            angles_dict.update({point_name: [angle, text_cords]})

    return angles_dict


def define_symmetry(angles_dict, allowed_diff=15, threshold=None):
    n1 = len([key for key in angles_dict.keys() if "right" in key])
    n2 = len([key for key in angles_dict.keys() if "left" in key])
    assert n1 == n2
    angles_to_analyze = ['_elbow', '_knee', '_shoulder', '_hip_center', '_neck_center', '_hip']
    asymmetry_dict = {}

    for angle in angles_to_analyze:
        right_point, left_point = 'right' + angle, 'left' + angle

        # get visibility for the points centers
        t1, t2 = float(angles_dict[right_point][1][-1]), float(angles_dict[left_point][1][-1])
        visibility = True if threshold is None else (t1 > threshold and t2 > threshold)

        if visibility:
            angle1 = angles_dict[right_point][0]
            angle2 = angles_dict[left_point][0]
            diff = angle1 - angle2

            if abs(diff) > allowed_diff:
                asymmetry_dict.update({angle: diff})

    return asymmetry_dict


def get_intersect(x1, y1, x2, y2):
    """
    Returns the point of intersection of the lines or None if lines are parallel
    Ex. p1=(x1,x2)... line_intersection((p1,p2), (p3,p4))
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([x1, y1, x2, y2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        return None, None
    return x / z, y / z


def get_line_coords(img_shape, p1, p2):
    """
    Returns the coordinates of a line passing through two specified points
    """
    x1, y1 = [int(p) for p in p1]
    x2, y2 = [int(p) for p in p2]
    div = 1.0 * (x2 - x1) if x2 != x1 else .00001
    a = (1.0 * (y2 - y1)) / div
    b = -a * x1 + y1
    y1_, y2_ = 0, img_shape[1]
    x1_ = int((y1_ - b) / a)
    x2_ = int((y2_ - b) / a)
    return (x1_, y1_), (x2_, y2_)


def triangle_centroid(p1, p2, p3):
    x = int((p1[0] + p2[0] + p3[0]) / 3)
    y = int((p1[1] + p2[1] + p3[1]) / 3)

    return x, y


def center_of_gravity(keypoints):
    # p1 = keypoints["left_ankle"]
    # p2 = keypoints["right_ankle"]
    # p3 = keypoints["neck_center"]
    # centroid = triangle_centroid(p1, p2, p3)
    x, y, _ = keypoints["hip_center"]
    x_line = (x, y)
    x1, y1, _ = keypoints["left_ankle"]
    x2, y2, _ = keypoints["right_ankle"]
    x = (x1 + x2) // 2
    y = (y1 + y2) // 2
    y_line = (x, y)

    return x_line, y_line


def get_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


if __name__ == "__main__":
    pass
