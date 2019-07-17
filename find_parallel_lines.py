import matplotlib.pyplot as plt
import matplotlib.image as pimg
import settings
import numpy as np
import math,cmath
import cv2
import pickle
from roi import roipoly
#images used to find the vanishing point
straight_images = ["test_images/test.jpg"]
roi = np.zeros((settings.ORIGINAL_SIZE[1], settings.ORIGINAL_SIZE[0]), dtype=np.uint8)
with open(settings.CALIB_FILE_NAME, 'rb') as f:
    calib_data = pickle.load(f)
    cam_matrix = calib_data["cam_matrix"]
    dist_coeffs = calib_data["dist_coeffs"]

Lhs = np.zeros((2,2), dtype= np.float32)
Rhs = np.zeros((2,1), dtype= np.float32)
def on_line(p1, p2, ycoord):
    return [p1[0]+ (p2[0]-p1[0])/float(p2[1]-p1[1])*(ycoord-p1[1]), ycoord]
def findparallel(lines):
    lines1 = []
    for i in range(len(lines)):
        for j in range(len(lines)):
            if (i == j):continue
            if (abs(lines[i][1] - lines[j][1]) == 0):
                 #You've found a parallel line!
                 lines1.append((i,j))
    return lines1

for img_path in straight_images:
    img = cv2.imread(img_path)
    cv2.imshow("1",img)
    cv2.waitKey(0)
    img = cv2.undistort(img, cam_matrix, dist_coeffs)
    cv2.imshow("2",img)
    cv2.waitKey(0)
    img_hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    cv2.imshow("3",img_hsl)
    cv2.waitKey(0)
    edges = cv2.Canny(img_hsl[:, :, 1], 200, 100)
    cv2.imshow("4",edges)
    cv2.waitKey(0)
    lines = cv2.HoughLinesP(edges*roi, 0.5, np.pi/180, 20, None, 180, 120)
    for line in lines:
        for x1, y1, x2, y2 in line:
            normal = np.array([[-(y2-y1)], [x2-x1]], dtype=np.float32)
            normal /=np.linalg.norm(normal)
            point = np.array([[x1],[y1]], dtype=np.float32)
            outer = np.matmul(normal, normal.T)
            Lhs += outer
            Rhs += np.matmul(outer, point)
            cv2.line(img, (x1,y1), (x2, y2),(255, 0, 0), thickness=2)

    cv2.imshow("sss",img)
    cv2.waitKey(0)
#     fig, ax = plt.subplots()
#     ax.imshow(img)
#     poly=roipoly(fig,ax)
#
# src_points=np.asarray(poly.getPolyVerts(),np.float32)
# print(src_points.shape)
# if src_points.shape[0]>=5:
#     src_points=src_points[0:4]
# cv2.polylines(img, [src_points.astype(np.int32)],True, (0,0,255), thickness=5)
# plt.imshow(img)
# plt.show()
# dst_points = np.array([[0, 0], [settings.UNWARPED_SIZE[0], 0],
#                        [settings.UNWARPED_SIZE[0], settings.UNWARPED_SIZE[1]],
#                        [0, settings.UNWARPED_SIZE[1]]], dtype=np.float32)
#
#
#
# # draw the trapezoid
#
#
# #find the projection matrix
# M = cv2.getPerspectiveTransform(src_points, dst_points)
# min_wid = settings.UNWARPED_SIZE[0]
#
#
# for img_path in straight_images:
#     img = pimg.imread(img_path)
#     img = cv2.undistort(img, cam_matrix, dist_coeffs)
#     img = cv2.warpPerspective(img, M, settings.UNWARPED_SIZE)
#     # img_hsl = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
#     # mask = img_hsl[:,:,1]>128
#     # mask[:, :50]=0
#     # mask[:, -50:]=0
#     # plt.imshow(mask)
#     # plt.show()
#     # mom = cv2.moments(mask[:,:settings.UNWARPED_SIZE[0]//2].astype(np.uint8))
#     # x1 = mom["m10"]/mom["m00"]
#     # mom = cv2.moments(mask[:,settings.UNWARPED_SIZE[0]//2:].astype(np.uint8))
#     # x2 = settings.UNWARPED_SIZE[0]//2 + mom["m10"]/mom["m00"]
#     # cv2.line(img, (int(x1), 0), (int(x1), settings.UNWARPED_SIZE[1]), (255, 0, 0), 3)
#     # cv2.line(img, (int(x2), 0), (int(x2), settings.UNWARPED_SIZE[1]), (0, 0, 255), 3)
#     # if (x2-x1<min_wid):
# #     #     min_wid = x2-x1
# # print(min_wid)
# pix_per_meter_x = min_wid/(6.09)
# Lh = np.linalg.inv(np.matmul(M, cam_matrix))
# pix_per_meter_y = pix_per_meter_x * np.linalg.norm(Lh[:,0]) / np.linalg.norm(Lh[:,1])
# print(pix_per_meter_x, pix_per_meter_y)
#
# fig, ax = plt.subplots()
# ax.imshow(img)
# poly=roipoly(fig,ax)
# line=np.asarray(poly.getPolyVerts(),np.float32)
# print("----------------------------------------------------")
# x=(abs(line[1][1]-line[0][1])/pix_per_meter_y)
# y=(abs(line[1][0]-line[0][0])/pix_per_meter_x)
# z=complex(x,y)
# print(abs(z))
# perspective_data = {'perspective_transform':M,
#               'pixels_per_meter':(pix_per_meter_x, pix_per_meter_y),
#               'orig_points':src_points}
# with open(settings.PERSPECTIVE_FILE_NAME, 'wb') as f:
#     pickle.dump(perspective_data, f)
#
