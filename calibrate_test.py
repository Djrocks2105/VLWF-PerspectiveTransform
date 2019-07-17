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
    with open(filename, 'rb') as f:
        calib_data = pickle.load(f)
        mtx = calib_data["cam_matrix"]
        dist = calib_data["dist_coeffs"]

    if not silent:
        for image_file in os.listdir(images_path):
            if image_file.endswith("jpg"):
                # show distorted images
                img = cv2.imread(os.path.join(images_path, image_file))
                for i in range(2):
                    img=cv2.pyrDown(img)
                plt.imshow(cv2.undistort(img, mtx, dist))
                plt.show()

    return mtx, dist

if __name__ == '__main__':
    calibrate(CALIB_FILE_NAME, False)








