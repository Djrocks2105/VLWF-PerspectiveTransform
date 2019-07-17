import cv2
import numpy as np

# img = cv2.imread()
img = cv2.imread("/home/jbmai/JBM/ALPR/outputs_old/3/85_11:25:00_output.png")
# # img = cv2.resize(img,(250,250))
yuv=cv2.GaussianBlur(img,(9,9),0)
# # cv2.imshow('detected circles',yuv )
# # cv2.waitKey(0)
yuv = cv2.cvtColor(img,cv2.COLOR_BGR2YUV)[:,:,0]
yuv = cv2.Canny(yuv,50,200)
cv2.imshow('detected circles',yuv)
cv2.waitKey(0)
#
# kernel = np.ones((3,3), np.uint8)
# yuv = cv2.dilate(yuv, kernel, iterations=4)
# # yuv = cv2.erode(yuv, kernel, iterations=2)
# retval, yuv = cv2.threshold(yuv, 150, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow('detected circles',yuv)
# cv2.waitKey(0)
# blobParams = cv2.SimpleBlobDetector_Params()
#
# # Change thresholds
#
# blobParams.minThreshold = 8
# blobParams.maxThreshold = 255
#
# # Filter by Area.
# blobParams.filterByArea = True
# blobParams.minArea = 64     # minArea may be adjusted to suit for your experiment
# blobParams.maxArea = 2500   # maxArea may be adjusted to suit for your experiment
#
# # Filter by Circularity
# blobParams.filterByCircularity = True
# blobParams.minCircularity = 0.1
#
# # Filter by Convexity
# blobParams.filterByConvexity = True
# blobParams.minConvexity = 0.87
#
# # Filter by Inertia
# blobParams.filterByInertia = True
# blobParams.minInertiaRatio = 0.01
# # Create a detector with the parameters
# blobDetector = cv2.SimpleBlobDetector_create(blobParams)
#
# keypoints = blobDetector.detect(img) # Detect blobs.
#
#             # Draw detected blobs as red circles. This helps cv2.findCirclesGrid() .
# im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
# cv2.imshow('detected circles',im_with_keypoints)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# # img = cv2.resize(img,(250,250))
# # yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)[:,:,0]
# # # cv2.resize((250,250))
# # kernel = np.ones((3,3), np.uint8)
# # # yuv=cv2.GaussianBlur(yuv,(3,3),0)
# # ret,yuv = cv2.threshold(yuv,0,50,cv2.THRESH_BINARY)
# # yuv = cv2.erode(yuv, kernel, iterations=1)
# # # yuv = cv2ptiveThreshold(yuv,105,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,2)
