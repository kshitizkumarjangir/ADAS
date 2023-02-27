import numpy as np


def get_veh_rel_dist(bbox):    # in meters
    # Image Size = (1920, 1080)

    # Exact distance
    # 3 mtrs - 983
    # 4 mtrs - 888
    # 5 mtrs - 815
    # 6 mtrs - 768
    # 7 mtrs - 733
    # 8 mtrs - 705
    # 9 mtrs - 684
    # 10 mtrs - 669
    # 11 mtrs - 659

    # approximation distance between exact values
    # in reality it follows perspective(angular) distance
    # warp image for exact distance

    lower_pt = bbox[3]

    if lower_pt >= 983:
        dist = 3
    elif 983 > lower_pt >= 888:
        d = (983 - lower_pt)/(983 - 888)     # approximation
        dist = 3 + d
    elif 888 > lower_pt >= 815:
        d = (888 - lower_pt)/(888 - 815)
        dist = 4 + d
    elif 815 > lower_pt >= 768:
        d = (815 - lower_pt)/(815 - 768)
        dist = 5 + d
    elif 768 > lower_pt >= 733:
        d = (768 - lower_pt)/(768 - 733)
        dist = 6 + d
    elif 733 > lower_pt >= 705:
        d = (733 - lower_pt)/(733 - 705)
        dist = 7 + d
    elif 705 > lower_pt >= 684:
        d = (705 - lower_pt)/(705 - 684)
        dist = 8 + d
    elif 684 > lower_pt >= 669:
        d = (684 - lower_pt)/(684 - 669)
        dist = 9 + d
    elif 669 > lower_pt >= 659:
        d = (669 - lower_pt)/(669 - 659)
        dist = 10 + d
    else:
        dist = 12

    return np.round(dist, 2)


def get_vehicle_relative_speed(pre_bbox, curr_bbox, timespan):  # in km/h
    # get vehicle distance
    pre_rel_dist = get_veh_rel_dist(pre_bbox)
    curr_rel_dist = get_veh_rel_dist(curr_bbox)

    if curr_rel_dist == 12 or curr_bbox[2] <= 250:
        return 999, 999  # out of scope/calibration

    # calculate speed
    speed = (18/5) * ((curr_rel_dist-pre_rel_dist)/timespan)

    # print('pre_rel_dist', pre_rel_dist)
    # print("curr_rel_dist ", curr_rel_dist)
    # print('timespan', timespan)
    # print('speed ', speed)

    return np.round(speed, 2), curr_rel_dist


