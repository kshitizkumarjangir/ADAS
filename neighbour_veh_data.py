import numpy as np


def get_veh_rel_dist(bbox):    # in meters
    # Image Size = (1920, 1080)
    # 3 mtrs - 983
    # 4 mtrs - 888
    # 5 mtrs - 815
    # 6 mtrs - 768
    # 7 mtrs - 733
    # 8 mtrs - 705
    # 9 mtrs - 684
    # 10 mtrs - 669
    # 11 mtrs - 659

    lower_pt = bbox[3]
    dist = 0
    if lower_pt >= 983:
        dist = 3
    elif 983 > lower_pt >= 888:
        dist = 4
    elif 888 > lower_pt >= 815:
        dist = 5
    elif 815 > lower_pt >= 768:
        dist = 6
    elif 768 > lower_pt >= 733:
        dist = 7
    elif 733 > lower_pt >= 705:
        dist = 8
    elif 705 > lower_pt >= 684:
        dist = 9
    elif 684 > lower_pt >= 669:
        dist = 10
    elif 669 > lower_pt >= 659:
        dist = 11
    else:
        dist = 12

    return np.round(dist, 0)


def get_vehicle_relative_speed(pre_bbox, curr_bbox, timespan):  # in km/h
    # get vehicle distance
    pre_rel_dist = get_veh_rel_dist(pre_bbox)
    curr_rel_dist = get_veh_rel_dist(curr_bbox)

    if curr_rel_dist == 12:
        return 999  # out of scope/calibration

    # calculate speed
    speed = 5/18 * (curr_rel_dist-pre_rel_dist)/timespan

    return np.round(speed, 2)


