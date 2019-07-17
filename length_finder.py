import matplotlib.pyplot as plt
import matplotlib.image as pimg
import settings
import numpy as np
import math,cmath
import cv2
import pickle
from roi import roipoly
import imutils
#images used to find the vanishing point
# straight_images = ["test_images/mask0.jpg","test_images/mask4.jpg"]
from numpy import random
from scipy.spatial import distance

def closest_node(node, nodes):
    closest_index = distance.cdist([node], nodes).argmin()
    return nodes[closest_index]
def findextremepoints(mask,src_points):
    # gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # gray=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    # gray = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow("sswd",gray)
    # cv2.waitKey(0)
    try:
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        # print(cnts)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts,key=cv2.contourArea)
        c_new=np.squeeze(c)
        # print(c_new)
        botleft=tuple(closest_node((settings.UNWARPED_SIZE[0]/2,settings.UNWARPED_SIZE[1]),c_new))
        topleft=tuple(closest_node((0,settings.UNWARPED_SIZE[1]),c_new))
        botright = tuple(closest_node((settings.UNWARPED_SIZE[0],settings.UNWARPED_SIZE[1]),c_new))
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        # extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        # cv2.drawContours(mask, [c], -1, (0, 255, 255), 2)
        print(cv2.pointPolygonTest(src_points,extLeft,False),cv2.pointPolygonTest(src_points,extRight,False),cv2.pointPolygonTest(src_points,extBot,False))
        if(cv2.pointPolygonTest(src_points,extLeft,False)<0) or (cv2.pointPolygonTest(src_points,extRight,False)<0)  or (cv2.pointPolygonTest(src_points,extBot,False)<0):
            return None,None,mask
        # cv2.circle(mask, botleft, 8, (0, 0, 0), -1)
        # cv2.circle(mask, topleft, 8, (255, 255, 0), -1)
        # cv2.circle(mask, botright, 8, (255, 0, 255), -1)
        # cv2.imshow("Image", mask)
        # cv2.waitKey(0)
        lengthpts=[topleft,botleft]
        widthpts=[botleft,botright]
        print(lengthpts,widthpts)
        return lengthpts,widthpts,mask
    except:
        return None,None,mask

def get_length_width(lengthpts,widthpts,M,pix_per_meter_x,pix_per_meter_y):
    lw=np.asarray([lengthpts,widthpts])
    lengthpts=lw[0]
    widthpts=lw[1]
    lengthpts[0]=[np.dot(M[0],np.append(lengthpts[0],[1]))/np.dot(M[2],np.append(lengthpts[0],[1])),np.dot(M[1],np.append(lengthpts[0],[1]))/np.dot(M[2],np.append(lengthpts[0],[1]))]
    lengthpts[1]=[np.dot(M[0],np.append(lengthpts[1],[1]))/np.dot(M[2],np.append(lengthpts[1],[1])),np.dot(M[1],np.append(lengthpts[1],[1]))/np.dot(M[2],np.append(lengthpts[1],[1]))]
    widthpts[0]=[np.dot(M[0],np.append(widthpts[0],[1]))/np.dot(M[2],np.append(widthpts[0],[1])),np.dot(M[1],np.append(widthpts[0],[1]))/np.dot(M[2],np.append(widthpts[0],[1]))]
    widthpts[1]=[np.dot(M[0],np.append(widthpts[1],[1]))/np.dot(M[2],np.append(widthpts[1],[1])),np.dot(M[1],np.append(widthpts[1],[1]))/np.dot(M[2],np.append(widthpts[1],[1]))]
    x=(abs(lengthpts[1][1]-lengthpts[0][1])/pix_per_meter_y)
    y=(abs(lengthpts[1][0]-lengthpts[0][0])/pix_per_meter_x)
    z=complex(x,y)
    length=(abs(z))
    x=(abs(widthpts[1][1]-widthpts[0][1])/pix_per_meter_y)
    y=(abs(widthpts[1][0]-widthpts[0][0])/pix_per_meter_x)
    z=complex(x,y)
    width=(abs(z))
    return length,width
def find_length(img):
    with open(settings.CALIB_FILE_NAME, 'rb') as f:
        calib_data = pickle.load(f,encoding='latin1')
        cam_matrix = calib_data["cam_matrix"]
        dist_coeffs = calib_data["dist_coeffs"]
    with open(settings.PERSPECTIVE_FILE_NAME, 'rb') as f:
        perspective_data = pickle.load(f,encoding='latin1')
        M = perspective_data["perspective_transform"]
        (pix_per_meter_x, pix_per_meter_y) = perspective_data["pixels_per_meter"]
        src_points=perspective_data["orig_points"]
    img = cv2.undistort(img, cam_matrix, dist_coeffs)
    # cv2.imshow("un")
    lengthpts,widthpts,img=findextremepoints(img,src_points)
    print(lengthpts,widthpts)
    if lengthpts is None :
        return None,None
    # img = cv2.warpPerspective(img, M, settings.UNWARPED_SIZE)
    length,width=get_length_width(lengthpts,widthpts,M,pix_per_meter_x,pix_per_meter_y)
    return length,width
if __name__ == "__main__":
    straight_images = ["test_images/mask0.jpg","test_images/mask1.jpg","test_images/mask2.jpg","test_images/mask3.jpg","test_images/mask4.jpg","test_images/mask5.jpg"]
    img=cv2.imread("test_images/mask5.jpg")
    x,y=find_length(img)
    print(x,y)
