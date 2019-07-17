"""
This script is used to calibrate the camera based on the provided images
The distortion coefficients and camera matrix are saved for later reuse
"""
import cv2
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
from settings import CALIB_FILE_NAME
from roi import roipoly

def calibrate(filename, silent = True):
    images_path = 'camera_cal'
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    ########################################Blob Detector##############################################

    # Setup SimpleBlobDetector parameters.
    blobParams = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    blobParams.minThreshold = 8
    blobParams.maxThreshold = 255

    # Filter by Area.
    blobParams.filterByArea = True
    blobParams.minArea = 64   # minArea may be adjusted to suit for your experiment
    blobParams.maxArea = 250000  # maxArea may be adjusted to suit for your experiment

    # Filter by Circularity
    blobParams.filterByCircularity = True
    blobParams.minCircularity = 0.1

    # Filter by Convexity
    blobParams.filterByConvexity = True
    blobParams.minConvexity = 0.87

    # Filter by Inertia
    blobParams.filterByInertia = True
    blobParams.minInertiaRatio = 0.01

    # Create a detector with the parameters
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)

    ###################################################################################################

    ###################################################################################################

    # Original blob coordinates, supposing all blobs are of z-coordinates 0
    # And, the distance between every two neighbour blob circle centers is 72 centimetres
    # In fact, any number can be used to replace 72.
    # Namely, the real size of the circle is pointless while calculating camera calibration parameters.
    objp = np.zeros((44, 3), np.float32)
    objp[0]  = (0  , 0  , 0)
    objp[1]  = (0  , 72 , 0)
    objp[2]  = (0  , 144, 0)
    objp[3]  = (0  , 216, 0)
    objp[4]  = (36 , 36 , 0)
    objp[5]  = (36 , 108, 0)
    objp[6]  = (36 , 180, 0)
    objp[7]  = (36 , 252, 0)
    objp[8]  = (72 , 0  , 0)
    objp[9]  = (72 , 72 , 0)
    objp[10] = (72 , 144, 0)
    objp[11] = (72 , 216, 0)
    objp[12] = (108, 36,  0)
    objp[13] = (108, 108, 0)
    objp[14] = (108, 180, 0)
    objp[15] = (108, 252, 0)
    objp[16] = (144, 0  , 0)
    objp[17] = (144, 72 , 0)
    objp[18] = (144, 144, 0)
    objp[19] = (144, 216, 0)
    objp[20] = (180, 36 , 0)
    objp[21] = (180, 108, 0)
    objp[22] = (180, 180, 0)
    objp[23] = (180, 252, 0)
    objp[24] = (216, 0  , 0)
    objp[25] = (216, 72 , 0)
    objp[26] = (216, 144, 0)
    objp[27] = (216, 216, 0)
    objp[28] = (252, 36 , 0)
    objp[29] = (252, 108, 0)
    objp[30] = (252, 180, 0)
    objp[31] = (252, 252, 0)
    objp[32] = (288, 0  , 0)
    objp[33] = (288, 72 , 0)
    objp[34] = (288, 144, 0)
    objp[35] = (288, 216, 0)
    objp[36] = (324, 36 , 0)
    objp[37] = (324, 108, 0)
    objp[38] = (324, 180, 0)
    objp[39] = (324, 252, 0)
    objp[40] = (360, 0  , 0)
    objp[41] = (360, 72 , 0)
    objp[42] = (360, 144, 0)
    objp[43] = (360, 216, 0)
    ###################################################################################################

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    # loop through provided images
    for image_file in os.listdir(images_path):
        if image_file.endswith("jpg"):
            # turn images to grayscale and find chessboard corners
            img = cv2.imread(os.path.join(images_path, image_file))
            # for i in range(2):
            #     img=cv2.pyrDown(img)
            #     print(img.shape)
            # cv2.imshow("wee",img)
            # cv2.waitKey(0)
            fig, ax = plt.subplots()
            ax.imshow(img)
            poly=roipoly(fig,ax)
            mask=poly.getMask(img)
            img=img.copy()
            img[~mask]=0
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plt.imshow(img_gray)
            plt.show()
            # img_gray=img_gray[mask]
            keypoints = blobDetector.detect(img_gray) # Detect blobs.
            # print(keypoints)
            # Draw detected blobs as red circles. This helps cv2.findCirclesGrid() .
            im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            cv2.namedWindow('image',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('image', 502, 376)
            cv2.imshow("image",im_with_keypoints)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            im_with_keypoints_gray = cv2.cvtColor(im_with_keypoints, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findCirclesGrid(im_with_keypoints, (4,11), None,flags=(cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING ) ,blobDetector=blobDetector)   # Find the circle grid
            print(ret)
            if ret == True:
                print(image_file)
                objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.
                corners2 = cv2.cornerSubPix(im_with_keypoints_gray, corners, (11,11), (-1,-1), criteria)    # Refines the corner locations.
                imgpoints.append(corners2)
                # Draw and display the corners.
                im_with_keypoints = cv2.drawChessboardCorners(img, (4,11), corners2, ret)
                plt.imshow(im_with_keypoints)
                plt.show()
            # else:
                # os.remove(images_path+"/"+image_file)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_gray.shape[::-1], None, None)
    # print(rvecs)
    # print(tvecs)
    img_size  = img.shape
    # rotation_mat = np.zeros(shape=(3, 3))
    # R = cv2.Rodrigues(rvecs[0], rotation_mat)[0]
    # P = np.column_stack((np.matmul(mtx,R), tvecs[0]))
    # print(P)
    #pickle the data and save it
    calib_data = {'cam_matrix':mtx,
                  'dist_coeffs':dist,
                  'img_size':img_size}
    with open(filename, 'wb') as f:
        pickle.dump(calib_data, f)

    if not silent:
        for image_file in os.listdir(images_path):
            if image_file.endswith("jpg"):
                # show distorted images
                img = cv2.imread(os.path.join(images_path, image_file))
                # for i in range(2):
                #     img=cv2.pyrDown(img)
                plt.imshow(cv2.undistort(img, mtx, dist))
                plt.show()

    return mtx, dist

if __name__ == '__main__':
    calibrate(CALIB_FILE_NAME, False)








