# Getting Projection Matrix And Camera Matrix For Finding Length And Width Of Car

The goals / steps of this project were the following:

* Compute the camera calibration matrix and distortion coefficients given a set of asymmetric circles images.
* Apply a distortion correction to raw images.
* Apply a perspective transform to rectify binary image ("birds-eye view") with the help of ROI extractor in file **roi.py**.
* Detect lane pixels and fit to find the lane boundary.
* Warp the detected lane boundaries back onto the original image.
* Get length and width of the vehicle in the warped image by finding extreme point you can find it in file **length_finder.py** .

[//]: # (Image References)

[image1]: ./readme_images/calibration.jpg "Original image"
[image2]: ./readme_images/corners.jpg "Image with corners"
[image3]: ./readme_images/undistorted.jpg "Undistorted image"
[image4]: ./readme_images/pinhole.png "Undistorted image"
[image10]:./readme_images/trapezoid.jpg "Image with trapezoid and vanishing point"
[image11]:./readme_images/warped.jpg "Warped original image"
[image14]: ./readme_images/undistorted_cf.png "Warp Example"
[image15]: ./readme_images/warped_cf.png "Fit Visual"
[image16]: ./test_images/test5.jpg "Initial image"
[image17]: ./readme_images/undistorted_si.jpg "Undistorted single image"
[image18]: ./readme_images/warped_si.jpg "Warped"
[image19]: ./readme_images/llab.jpg "Undistorted single image"
[image20]: ./readme_images/lhls.jpg "Warped"
[image21]: ./readme_images/blab.jpg "Initial image"
[image22]: ./readme_images/llab_thresh.jpg "Undistorted single image"
[image23]: ./readme_images/lhls_thresh.jpg "Warped"
[image24]: ./readme_images/blab_thresh.jpg "Initial image"
[image25]: ./readme_images/total_mask.jpg "Undistorted single image"
[image26]: ./readme_images/eroded_mask.jpg "Undistorted single image"
[image27]: ./readme_images/line_masks.jpg "Undistorted single image"
[image28]: ./readme_images/lines.jpg "Undistorted single image"
[image29]: ./readme_images/warped_lanes.jpg "Undistorted single image"
[image30]: ./output_images/test5.jpg "Undistorted single image"

## Camera Calibration
Before starting the implementation of the lane detection pipeline, the first thing that should be done is camera calibration. That would help us:
* undistort the images coming from camera, thus improving the quality of geometrical measurement
* estimate the spatial resolution of pixels per meter in both *x* and *y* directions

For this project, the calibration pipeline is implemented in function `calibrate` implemented in file `calibrate.py`. The procedure follows these steps:

The procedures start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0. Also,  I am assuming that the size of the chessboard pattern is the same for all images, so the object points will be the same for each calibration image.

Then, the calibration images are sequentially loaded, converted to grayscale and chessboard pattern is looked for using `cv2.findChessboardCorners`. When the pattern is found, the positions of the corners get refined further to sub-pixel accuracy using `cv2.cornerSubPix`. That improves the accuracy of the calibration. Corner coordinates get appended to the list containing all the image points, while prepared "object points" gets appended to the list containing all the object points. 

The distortion coefficients and camera matrix are computed using `cv2.calibrateCamera()` function, where image and object points are passed as inputs. It is important to check if the result is satisfactory, since calibration is a nonlinear numerical procedure, so it might yield suboptimal results. To do so calibration images are read once again and undistortion is applied to them. The undistorted images are shown in order to visually check if the distortion has been corrected. Once the data has been checked the parameters are is pickled saved to file. One sample of the input image, image with chessboard corners and the undistorted image is shown:

| Original image     | Chessboard corners | Undistorted image  |
|--------------------|--------------------|--------------------|
|![alt text][image1] |![alt text][image2] |![alt text][image3] |

Before we move further on, lets just reflect on what the camera matrix is. The camera matrix encompases the pinhole camera model in it. It gives the relationship between the coordinates of the points relative to the camera in 3D space and position of that point on the image in pixels. If *X*, *Y* and *Z* are coordinates of the point in 3D space, its position on image (*u* and *v*) in pixels is calculated using:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{bmatrix}u\\v\\1\end{bmatrix}=s\mathbf{M}\begin{bmatrix}X\\Y\\Z\end{bmatrix}" alt="{https://latex.codecogs.com/svg.latex?\begin{bmatrix}u\\v\\1\end{bmatrix}=s\mathbf{M}\begin{bmatrix}X\\Y\\Z\end{bmatrix}}">
</p>

where **M** is camera matrix and *s* is scalar different from zero. This equation will be used later on.

| Pinhole camera model          |
|-------------------------------|
|![Pinhole camera model][image4]|


## Finding projective transformation
Next step is to find the projective transform so the original images can be warped so that it looks like the camera is placed directly above the road. Our approach is to hand tune the source and destination points, which are required to compute the transformation matrix. On the other hand, the script that does that for us can be created based on linear perspective geometry.

Here we select the corners of the trapezoid manually with the help of an ROI extractor which is used in **find_perspective_transform.py**

The corners of the trapezoid are used as a source points, while destination points are four corners of the new image. The size of the warped image is defined in file `settings.py`. After that, the matrix that defines the perspective transform is calculated using `cv2.getPerspectiveTransform()`. The procedure that implements the calculation of homography matrix of the perspective transform is implemented at the beginning of the python script `find_perspective_transform.py`. Images that illustrate the procedure follow. 

|Trapezoid                    |     Warped image   |
|-----------------------------|--------------------|
|![alt text][image10]         |![alt text][image11]|

The obtained source and destination points are:

| Source        | Destination   |
|:-------------:|:-------------:|
| 375, 480      | 0, 0          |
| 905, 480      | 500, 0        |
| 1811, 685     | 500, 600      |
| -531, 685     | 0, 600        |


Once again, lets just reflect on what is the matrix returned by the `cv2.getPerspectiveTransform()`. It tells how the perspective transformation is going to be performed and where the pixel from original image  with the coordinates (*u*, *v*) is going to move after the transformation. The destination of that pixel on the warped image would be the point with the coordinates (*u<sub>w</sub>*, *v<sub>w</sub>*). The new position is calculated using::

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{bmatrix}u_w\\v_w\\1\end{bmatrix}=s\mathbf{H}\begin{bmatrix}u\\v\\1\end{bmatrix}" alt="{mathcode}">
</p>

where **H** is homography matrix returned as a result of `cv2.getPerspectiveTransform()` and *s* is scalar different from zero. 


## Estimating pixel resolution
Next important step is to estimate resolution in pixels per meter of the warped image. It also can be done by hand, but as previously we'll create a script that does that for us. In the course materials, it was stated that width of the road is  6.02 meters.In order to estimate the resolution in pixels per meter, the images with the road without vehicles  will be used which is in ***test_images*** folder. They will be unwarped and the distance of the road will be measured.

Since we now know width of the road in pixels is the width of the warped image. That allows for the calculation of the width in pixels and then resolution in *x* direction. This procedure is implemented in the script `find_perspective_transform.py`. The images that illustrate the procedure are shown below.

That is how to find resolution in *x* direction, but for finding resolution in *y* there is no such trick. Since nothing was estimated manually neither this will be. The camera matrix obtained by calibrations holds some relative information about resolutions in *x* and *y* direction. We can try to exploit that. To find resolution in *y* direction we have to do some more math. 

Lets say, we have a coordinate frame attached to the road, as shown on image below. It is easy to see that transformation of the coordinates represented in the coordinate frame of the road to the coordinate frame of the warped image, consists of scaling and shifted. Scale in *x* direction corresponds to the number of pixel per meter in *x* direction. Same holds for *y*. In mathemathical form that can be written as:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{bmatrix}u_w\\v_w\\1\end{bmatrix}=\begin{bmatrix}r_x&0&c_x\\0&r_y&c_y\\0&0&1\end{bmatrix}\begin{bmatrix}X_r\\Y_r\\1\end{bmatrix}" alt="{mathcode}">
</p>

| Road frame as seen by camera| Road frame on warped image |
|-----------------------------|----------------------------|
|![alt text][image14]         |![alt text][image15]        |


The same thing can be calculated from the other side. Lets say that position and of the road coordinate frame in camera coordinate frame is given with rotation matrix **R**=*[r<sub>1</sub> r<sub>2</sub> r<sub>3</sub>]* and translation vector *t*. One important property that is going to be exploited is that matrix **R** is orthogonal, meaning that each of the rows *r<sub>1</sub>, r<sub>2</sub>, r<sub>3</sub>* has the length of 1. Now since we know that, the pixel on the image that corresponds to the point with coordinates *X<sub>r</sub>, Y<sub>r</sub>* and *Z<sub>r</sub>* is calculated by:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{bmatrix}u\\v\\1\end{bmatrix}=s\mathbf{M}\left[r_1\;r_2\;r_3\;t\right]\begin{bmatrix}X_r\\Y_r\\Z_r\\1\end{bmatrix}" alt="{mathcode}">
</p>

Since the road is planar, the *Z<sub>r</sub>=0*. Now we apply the perspective transform and get:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?\begin{bmatrix}u_w\\v_w\\1\end{bmatrix}=s\mathbf{H}\mathbf{M}\left[r_1\;r_2\;t\right]\begin{bmatrix}X_r\\Y_r\\1\end{bmatrix}=\begin{bmatrix}r_x&0&c_x\\0&r_y&c_y\\0&0&1\end{bmatrix}\begin{bmatrix}X_r\\Y_r\\1\end{bmatrix}" alt="{mathcode}">
</p>
Since it has to hold for every point we can conclude that:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?s\left[r_1\;r_2\;t\right]=(\mathbf{H}\mathbf{M})^{-1}\left\begin{bmatrix}r_x&0&c_x\\0&r_y&c_y\\0&0&1\end{bmatrix}=[h_1\;h_2\;h_3]\begin{bmatrix}r_x&0&c_x\\0&r_y&c_y\\0&0&1\end{bmatrix}" alt="{mathcode}">
</p>

Where *h<sub>1</sub>, h<sub>2</sub>, h<sub>3</sub>* are columns of matrix (**HM**)*<sup>-1*. Since the length of vectors *r<sub>1</sub>* and *r<sub>2</sub>* is one, we can calculate scalar *s* and finally *r<sub>y</sub>*:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?r_y=r_x\frac{\left||h_1\right||}{\left||h_2\right||}" alt="{mathcode}">
</p>

The rather simple equation is obtained at the end, but it took us a while to get there. This calculation is implemented in lines 91 and 92 of script `find_perspective_transform.py`. The final result is:


| pix/meter in *x* direction | pix/meter in *y* direction | 
|----------------------------|----------------------------|
|46.567                         |      33.0652548749         |

## The Length Finder
This file has a function **find_length** which we will use .This function takes an mask of a car from the camera image and remove the distortion and give us the length and width if it is in right frame.
The function **findextremepoints** is used to find the points between which we can get length and width of the car.Also this function checks whether the mask of the car is good enough to get he length.
The function **get_length_width** gives us the length and width in meters.
Here we are finding the points by calculating the points on the car mask contour which are closest to the bottom left of the image,bottom center and bottom right.
You may need to change this points according to the orientation of your camera in real world which can be done in **findextremepoints** 
We will be using only the projection.p and calibration.p with **length_finder.py**  and **setting.py** for the masks returend from mask-RCNN to get the length and width of car you can see some of them in test_images folder and run some test on them by running the following script:
```shellscript
    $ python length_finder.py
```
Be sure to change the projection and calibration file to your created one in **setting.py**
Also be sure to change the blob area according to your needs of experiments.
