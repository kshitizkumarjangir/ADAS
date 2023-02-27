import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


def get_distortion_factors():
    # Prepare object points
    # From the provided calibration images, 9*6 corners are identified
    objpoints = []
    imgpoints = []
    # Object points are real world points, here a 3D coordinates matrix is generated
    # z coordinates are 0 and x, y are equidistant as it is known that the chessboard is made of identical squares
    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Make a list of calibration images
    os.listdir("camera_calibration/")
    cal_img_list = os.listdir("camera_calibration/")

    # Image-points are the correspondent object points with their coordinates in the distorted image
    for image_name in cal_img_list:
        import_from = 'camera_calibration/' + image_name
        img = cv2.imread(import_from)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # if ret:
        # corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)

        # If found, draw corners
        if ret:
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)  #
            imgpoints.append(corners)
            objpoints.append(objp)

    cv2.waitKey(0)

    dim = gray.shape[::-1]
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objectPoints=objpoints,
                                                       imagePoints=imgpoints,
                                                       imageSize=dim,
                                                       cameraMatrix=None,
                                                       distCoeffs=None)
    print('re-projection error', ret)
    return mtx, dist


def warp(img, mtx, dist):
    # get un-distort image
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    img_size = (img.shape[1], img.shape[0])
    offset = 150

    # Source points taken from images with straight lane lines,
    # these are to become parallel after the warp transform
    src = np.float32([
        (460, 970),  # bottom-left corner
        (978, 660),  # top-left corner
        (1263, 660),  # top-right corner
        (1768, 970)  # bottom-right corner
    ])
    # src = np.float32([
    #     (750, 990),  # bottom-left corner
    #     (750, 650),  # top-left corner
    #     (1400, 650),  # top-right corner
    #     (1400, 990)  # bottom-right corner
    # ])
    # Destination points are to be parallel, taken into account the image size
    dst = np.float32([
        [offset, img_size[1]],  # bottom-left corner
        [offset, 0],  # top-left corner
        [img_size[0] - offset, 0],  # top-right corner
        [img_size[0] - offset, img_size[1]]  # bottom-right corner
    ])
    # Calculate the transformation matrix and it's inverse transformation
    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(undist, M, img_size)

    warped = cv2.resize(warped, (640, 640))
    cv2.imshow("img", warped)  # Display wrapped image
    cv2.waitKey(0)

    return warped, M_inv, undist


def main():
    # Get distortion factors
    mtx, dist = get_distortion_factors()

    img = cv2.imread("testdata/video/highway/ReferenceImage.png")
    warped, M_inv, undist = warp(img, mtx, dist)


if __name__ == "__main__":
    nx = 9  # The number of inside corners in x
    ny = 6  # The number of inside corners in y
    main()
