import matplotlib

import torch
import cv2
import numpy as np
import time
import colorsys
import matplotlib

# import deep sort libraries
from deep_sort_realtime.deepsort_tracker import DeepSort
import neighbour_veh_data as nvd


def get_closest_color_and_status(bgr):
    colors = ((0, 0, 255),  # red color
              (0, 255, 165),  # orange color
              (0, 255, 0))  # green color

    euclidian_dist = []
    for color in colors:
        dist = np.sqrt(np.square(bgr[0] - color[0]) + np.square(bgr[1] - color[1]) + np.square(bgr[2] - color[2]))
        euclidian_dist.append([dist, color])

    closest_color = min(euclidian_dist)[1]

    h, s, v = colorsys.rgb_to_hsv(bgr[2] / 255, bgr[1] / 255, bgr[0] / 255)
    print("closest color: ", closest_color, " h: ", h, " v:", v)

    status = ''
    if closest_color == colors[0]:
        status = 'STOP'
    elif closest_color == colors[1]:
        status = 'SLOW DOWN'
    else:
        status = 'GO'

    return status, closest_color


def traffic_signal_caution(frame, bbox_traffic):
    # reduce the size to crop - focus only on lights
    if squeez_traffic_light_factor < 1:
        f = 1 - squeez_traffic_light_factor
        h = int(bbox_traffic[3]) - int(bbox_traffic[1])
        w = int(bbox_traffic[2]) - int(bbox_traffic[0])

        if h > w:
            h_f = int((h * 0.1) / 2)
            w_f = int((w * f) / 2)
        else:
            w_f = int((w * 0.1) / 2)
            h_f = int((h * f) / 2)

        bbox_traffic[3] = int(bbox_traffic[3]) - h_f
        bbox_traffic[1] = int(bbox_traffic[1]) + h_f
        bbox_traffic[2] = int(bbox_traffic[2]) - w_f
        bbox_traffic[0] = int(bbox_traffic[0]) + w_f

    # Crop traffic signal image
    signal_img = frame[int(bbox_traffic[1]):int(bbox_traffic[3]), int(bbox_traffic[0]):int(bbox_traffic[2])]

    # apply Gaussian blur
    signal_img = cv2.GaussianBlur(signal_img, (gaussian_blur_rad, gaussian_blur_rad), 0)

    org_img = signal_img.copy()

    # convert to gray scale
    signal_img = cv2.cvtColor(signal_img, cv2.COLOR_BGR2GRAY)

    # find the brightest pixel location
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(signal_img)

    # find the color of the brightest point
    brightest_color = org_img[maxLoc[1], maxLoc[0]]

    h, s, v = colorsys.rgb_to_hsv(brightest_color[2] / 255, brightest_color[1] / 255, brightest_color[0] / 255)
    if h > 0:
        h = np.round((360 * h), 0)
    if v > 0:
        v = np.round(v, 2)
    # status, color = get_closest_color_and_status(brightest_color)

    return v, h


def is_pt_on_track(pts):
    # frontal area of the car
    poly = np.array([
        (460, 970),  # bottom-left corner
        (978, 660),  # top-left corner
        (1263, 660),  # top-right corner
        (1768, 970)  # bottom-right corner
    ])

    path = matplotlib.path.Path(poly)

    if pts[1] > 700:
        test = 11

    is_inside = path.contains_point(pts)

    return is_inside


def detectObject(video_feed, object_type_to_tracked, min_confidence=0.49):
    # Generate random color
    rand_color = list(np.random.choice(range(255), size=(80, 3), replace=False))

    # Load Yolov5 model
    model = torch.hub.load('yolov5', 'yolov5m', source='local')
    class_list = model.names  # class name dict

    # read webcam feed
    cap = cv2.VideoCapture(video_feed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    object_tracker = DeepSort(max_age=10,
                              n_init=2,
                              nms_max_overlap=0.8,
                              max_cosine_distance=0.3,
                              nn_budget=None,
                              override_track_class=None,
                              embedder="mobilenet",
                              half=True,
                              bgr=True,
                              embedder_gpu=True,
                              embedder_model_name=None,
                              embedder_wts=None,
                              polygon=False,
                              today=None)

    pre_bbox = {}
    fps = 0
    veh_dir_data = {}

    while True:
        _, img = cap.read()
        if img is None:
            break

        t1 = time.time()
        # img = cv2.resize(img, (640, 480))

        # Detect frame image - Yolo
        result = model(img)

        # Read detected objects frame data
        df = result.pandas().xyxy[0]

        detections = []
        for idx in df.index:
            class_name = df['name'][idx]
            confidence = (df['confidence'][idx]).round(2)

            if confidence > min_confidence:  # check for certainty
                if class_name in object_type_to_tracked or len(object_type_to_tracked) == 0:
                    x1, y1 = int(df['xmin'][idx]), int(df['ymin'][idx])
                    x2, y2 = int(df['xmax'][idx]), int(df['ymax'][idx])
                    # class_idx = df['class'][idx]
                    # obj_text = class_name + " " + str(confidence)
                    # obj_color = tuple(list(rand_color[int(class_idx)]))
                    detections.append(([x1, y1, int(x2 - x1), int(y2 - y1)], confidence, class_name))

        is_traffic_light_present = False
        traffic_vh = []

        if is_tracking:

            # Tracking Objects
            tracks = object_tracker.update_tracks(detections, frame=img)

            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue
                track_id = track.track_id
                bbox = track.to_ltrb()
                class_name = track.get_det_class()

                obj_color = rand_color[list(class_list.values()).index(class_name)]
                obj_color = tuple(np.ndarray.tolist(obj_color))

                # Update Caution Status for traffic light
                if class_name == 'traffic light':
                    v, h = traffic_signal_caution(img, bbox)
                    is_traffic_light_present = True
                    traffic_vh.append([v, h])

                identifier = class_name + " ID: " + str(track_id)

                # Add marker and class name + track ID
                cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), obj_color, 2)
                # cv2.putText(img, identifier, (int(bbox[0]), int(bbox[1] - 10)),
                #            cv2.FONT_HERSHEY_PLAIN, 1, obj_color, 2)

                # add vehicle live data
                if fps != 0 and class_name in ['car', 'truck', 'bus', 'bicycle', 'motorcycle']:

                    if detect_speed_distance:
                        # Add relative speed and distance
                        if identifier in pre_bbox.keys():
                            speed = nvd.get_vehicle_relative_speed(pre_bbox[identifier], bbox, 1 / fps)
                            rel_distance = nvd.get_veh_rel_dist(bbox)

                            # print(identifier + ' speed: ', speed)
                            # print(identifier + ' dist: ', rel_distance)

                            if speed != 999: # check for in-scope
                                cv2.putText(img, "speed: " + str(speed), (int(bbox[0]), int(bbox[1] + 10)),
                                            cv2.FONT_HERSHEY_PLAIN, 2, obj_color, 2)
                                cv2.putText(img, "distance: " + str(rel_distance), (int(bbox[0]), int(bbox[1] + 25)),
                                            cv2.FONT_HERSHEY_PLAIN, 2, obj_color, 2)
                        else:
                            pre_bbox[identifier] = bbox

                    # Add predicted vehicle direction
                    if predict_direction:
                        # check side (LHS/RHS) of veh
                        if abs(bbox[0] - veh_center_line_width) < abs(bbox[2] - veh_center_line_width):
                            veh_pt = (int(bbox[0]), int(bbox[3]))
                        else:
                            veh_pt = (int(bbox[2]), int(bbox[3]))

                        if identifier in veh_dir_data.keys():
                            if veh_dir_data[identifier][0] >= update_veh_move_status_after_frame:
                                # Check veh position, if in front of my car
                                on_track = is_pt_on_track(veh_pt)
                                cv2.circle(img, veh_pt, 4, (255, 0, 0), -1)

                                warning_pt = (int((bbox[0] + bbox[2]) / 2 - 5), int(bbox[1] - 15))

                                if veh_dir_data[identifier][1][0] < veh_pt[0]:
                                    if on_track:
                                        cv2.putText(img, ">>>", warning_pt,
                                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                                    else:
                                        cv2.putText(img, ">>>", warning_pt,
                                                    cv2.FONT_HERSHEY_PLAIN, 1, (80, 205, 250), 2)
                                elif veh_dir_data[identifier][1][0] > veh_pt[0]:
                                    if on_track:
                                        cv2.putText(img, "<<<", warning_pt,
                                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                                    else:
                                        cv2.putText(img, "<<<", warning_pt,
                                                    cv2.FONT_HERSHEY_PLAIN, 1, (80, 205, 250), 2)
                                else:
                                    if on_track:
                                        cv2.putText(img, "OOO", warning_pt,
                                                    cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
                                    else:
                                        cv2.putText(img, "OOO", warning_pt,
                                                    cv2.FONT_HERSHEY_PLAIN, 1, (80, 205, 250), 2)

                                veh_dir_data[identifier] = [1, veh_pt]
                            else:
                                veh_dir_data[identifier][0] += 1
                        else:
                            veh_dir_data[identifier] = [1, veh_pt]

        else:
            for d in detections:
                class_name1 = d[2]
                bbox1 = [d[0][0], d[0][1], d[0][2] + d[0][0], d[0][3] + d[0][1]]
                confidence1 = np.round(d[1], 2)

                # Update Caution Status for traffic light
                if class_name1 == 'traffic light':
                    v, h = traffic_signal_caution(img, bbox1)
                    is_traffic_light_present = True
                    traffic_vh.append([v, h])

                obj_color1 = rand_color[list(class_list.values()).index(class_name1)]
                obj_color1 = tuple(np.ndarray.tolist(obj_color1))

                # Add marker and class name + track ID
                cv2.rectangle(img, (int(bbox1[0]), int(bbox1[1])), (int(bbox1[2]), int(bbox1[3])), obj_color1, 2)
                cv2.putText(img, class_name1 + " - " + str(confidence1), (int(bbox1[0]), int(bbox1[1] - 10)),
                            cv2.FONT_HERSHEY_PLAIN, 1, obj_color1, 2)

        # In case of multiple traffic light, focus only on switched on light
        if is_traffic_light_present:
            brightest_color = max(traffic_vh)

            signal_status = 'STOP'
            textcolor = (0, 0, 255)

            if brightest_color[0] >= min_traffic_light_brightness:  # check for switch on

                if brightest_color[0] > 0.9:  # very sure case
                    if 115 <= brightest_color[1] <= 295:
                        signal_status = 'GO'
                        textcolor = (0, 255, 0)

                elif 90 <= brightest_color[1] <= 195:
                    signal_status = 'GO'
                    textcolor = (0, 255, 0)

                print("selected color: ", brightest_color)
                cv2.putText(img, "Traffic Signal: " + signal_status, org=(0, 60),
                            fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=0.9,
                            color=textcolor, thickness=2)
            else:
                print("Current Traffic light: ", brightest_color)

        # Fetch FPS
        fps = 1. / (time.time() - t1)
        cv2.putText(img, "FPS: {:.0f}".format(fps), org=(0, 30), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=3,
                    color=(255, 255, 0), thickness=2)

        # img = cv2.resize(img, (1080, 720))
        cv2.imshow("live_video", img)

        c = cv2.waitKey(1)
        if c == 27:  # Stop when Esc is pressed
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # ------ Control parameters

    # track objects
    is_tracking = True

    # detect traffic lights status
    detect_traffic_light_status = True
    min_traffic_light_brightness = 0.7
    squeez_traffic_light_factor = 0.5  # to reduce extra area/cover around the light
    gaussian_blur_rad = 3

    # detect speed and distance
    detect_speed_distance = False

    # predict direction of movement
    predict_direction = True
    update_veh_move_status_after_frame = 10
    veh_center_line_width = int((460 + 1768) / 2)

    videoPath = 'testdata/video/highway/'
    videoFile = 'IMG_1828.MOV'
    object_type_to_tracked_1 = ['car', 'truck', 'bus', 'bicycle', 'person', 'motorcycle', 'traffic light']
    min_confidence_1 = 0.49
    detectObject(videoPath + videoFile, object_type_to_tracked_1, min_confidence_1)
    # detectObject(0, object_type_to_tracked_1, min_confidence_1)
