# imports
import os,sys
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from matplotlib import pyplot as plt

# add base directory to sys.path
base_dir = r'C:\Users\User\RaspberryPi4_PCfolder\repo_learningopenCV\chapter10'
sys.path.append(os.path.abspath(base_dir))


# Load stereo images
imgL = cv2.imread(r'C:\Users\User\RaspberryPi4_PCfolder\flowerL.jpeg',cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(r'C:\Users\User\RaspberryPi4_PCfolder\flowerR.jpeg',cv2.IMREAD_GRAYSCALE)
imgL = cv2.resize(imgL,(455,455), interpolation = cv2.INTER_AREA)
imgR = cv2.resize(imgR,(455,455), interpolation = cv2.INTER_AREA)

 
# Reading the mapping values for stereo image rectification
cv_file = cv2.FileStorage("data/stereo_rectify_maps.xml", cv2.FILE_STORAGE_READ)
Left_Stereo_Map_x = cv_file.getNode("Left_Stereo_Map_x").mat()
Left_Stereo_Map_y = cv_file.getNode("Left_Stereo_Map_y").mat()
Right_Stereo_Map_x = cv_file.getNode("Right_Stereo_Map_x").mat()
Right_Stereo_Map_y = cv_file.getNode("Right_Stereo_Map_y").mat()
cv_file.release()
 
def nothing(x):
    pass
 
cv2.namedWindow('disp',cv2.WINDOW_NORMAL)
cv2.resizeWindow('disp',600,600)
 
cv2.createTrackbar('numDisparities','disp',1,17,nothing)
cv2.createTrackbar('blockSize','disp',5,50,nothing)
cv2.createTrackbar('preFilterType','disp',1,1,nothing)
cv2.createTrackbar('preFilterSize','disp',2,25,nothing)
cv2.createTrackbar('preFilterCap','disp',5,62,nothing)
cv2.createTrackbar('textureThreshold','disp',10,100,nothing)
cv2.createTrackbar('uniquenessRatio','disp',15,100,nothing)
cv2.createTrackbar('speckleRange','disp',0,100,nothing)
cv2.createTrackbar('speckleWindowSize','disp',3,25,nothing)
cv2.createTrackbar('disp12MaxDiff','disp',5,25,nothing)
cv2.createTrackbar('minDisparity','disp',5,25,nothing)
 
# Creating an object of StereoBM algorithm
stereo = cv2.StereoBM_create()
 
while True:
 
  # Capturing and storing left and right camera images
  retL, imgL= CamL.read()
  retR, imgR= CamR.read()
   
  # Proceed only if the frames have been captured
  if retL and retR:
    imgR_gray = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)
    imgL_gray = cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
 
    # Applying stereo image rectification on the left image
    Left_nice= cv2.remap(imgL_gray,
              Left_Stereo_Map_x,
              Left_Stereo_Map_y,
              cv2.INTER_LANCZOS4,
              cv2.BORDER_CONSTANT,
              0)
     
    # Applying stereo image rectification on the right image
    Right_nice= cv2.remap(imgR_gray,
              Right_Stereo_Map_x,
              Right_Stereo_Map_y,
              cv2.INTER_LANCZOS4,
              cv2.BORDER_CONSTANT,
              0)
 
    # Updating the parameters based on the trackbar positions
    numDisparities = cv2.getTrackbarPos('numDisparities','disp')*16
    blockSize = cv2.getTrackbarPos('blockSize','disp')*2 + 5
    preFilterType = cv2.getTrackbarPos('preFilterType','disp')
    preFilterSize = cv2.getTrackbarPos('preFilterSize','disp')*2 + 5
    preFilterCap = cv2.getTrackbarPos('preFilterCap','disp')
    textureThreshold = cv2.getTrackbarPos('textureThreshold','disp')
    uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio','disp')
    speckleRange = cv2.getTrackbarPos('speckleRange','disp')
    speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize','disp')*2
    disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff','disp')
    minDisparity = cv2.getTrackbarPos('minDisparity','disp')
     
    # Setting the updated parameters before computing disparity map
    stereo.setNumDisparities(numDisparities)
    stereo.setBlockSize(blockSize)
    stereo.setPreFilterType(preFilterType)
    stereo.setPreFilterSize(preFilterSize)
    stereo.setPreFilterCap(preFilterCap)
    stereo.setTextureThreshold(textureThreshold)
    stereo.setUniquenessRatio(uniquenessRatio)
    stereo.setSpeckleRange(speckleRange)
    stereo.setSpeckleWindowSize(speckleWindowSize)
    stereo.setDisp12MaxDiff(disp12MaxDiff)
    stereo.setMinDisparity(minDisparity)
 
    # Calculating disparity using the StereoBM algorithm
    disparity = stereo.compute(Left_nice,Right_nice)
    # NOTE: Code returns a 16bit signed single channel image,
    # CV_16S containing a disparity map scaled by 16. Hence it 
    # is essential to convert it to CV_32F and scale it down 16 times.
 
    # Converting to float32 
    disparity = disparity.astype(np.float32)
 
    # Scaling down the disparity values and normalizing them 
    disparity = (disparity/16.0 - minDisparity)/numDisparities
 
    # Displaying the disparity map
    cv2.imshow("disp",disparity)
 
    # Close window using esc key
    if cv2.waitKey(1) == 27:
      break
   
  else:
    CamL= cv2.VideoCapture(CamL_id)
    CamR= cv2.VideoCapture(CamR_id)


# solving for M in the following equation
# ||    depth = M * (1/disparity)   ||
# for N data points coeff is Nx2 matrix with values 
# 1/disparity, 1
# and depth is Nx1 matrix with depth values
ret, sol = cv2.solve(coeff,z,flags=cv2.DECOMP_QR)    

depth_thresh = 100.0 # Threshold for SAFE distance (in cm)
 
# Mask to segment regions with depth less than threshold
mask = cv2.inRange(depth_map,10,depth_thresh)
 
# Check if a significantly large obstacle is present and filter out smaller noisy regions
if np.sum(mask)/255.0 > 0.01*mask.shape[0]*mask.shape[1]:
 
  # Contour detection 
  contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  cnts = sorted(contours, key=cv2.contourArea, reverse=True)
   
  # Check if detected contour is significantly large (to avoid multiple tiny regions)
  if cv2.contourArea(cnts[0]) > 0.01*mask.shape[0]*mask.shape[1]:
 
    x,y,w,h = cv2.boundingRect(cnts[0])
 
    # finding average depth of region represented by the largest contour 
    mask2 = np.zeros_like(mask)
    cv2.drawContours(mask2, cnts, 0, (255), -1)
 
    # Calculating the average depth of the object closer than the safe distance
    depth_mean, _ = cv2.meanStdDev(depth_map, mask=mask2)
     
    # Display warning text
    cv2.putText(output_canvas, "WARNING !", (x+5,y-40), 1, 2, (0,0,255), 2, 2)
    cv2.putText(output_canvas, "Object at", (x+5,y), 1, 2, (100,10,25), 2, 2)
    cv2.putText(output_canvas, "%.2f cm"%depth_mean, (x+5,y+40), 1, 2, (100,10,25), 2, 2)
 
else:
  cv2.putText(output_canvas, "SAFE!", (100,100),1,3,(0,255,0),2,3)
 
cv2.imshow('output_canvas',output_canvas)

# # # Rectify images
# # # TODO: Obtain camera intrinsic and extrinsic parameters for left and right cameras, as well as stereo calibration results
# # # Define the size of the chessboard pattern
#pattern_size = (9, 6)

# # # Define the object points of the pattern (assuming the pattern is fixed on a flat surface)
# # objectPoints = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
# # objectPoints[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# # # Initialize arrays to store object points and image points from all images
# # object_points_list = []
# # image_points_left_list = []
# # image_points_right_list = []


# # # Find chessboard corners in the images
# # ret_left, corners_left = cv2.findChessboardCorners(imgL, pattern_size)
# # ret_right, corners_right = cv2.findChessboardCorners(imgR, pattern_size)

# # # If corners are found in both images, add object points and image points to the lists
# # if ret_left and ret_right:
# #     object_points_list.append(objectPoints)
# #     image_points_left_list.append(corners_left)
# #     image_points_right_list.append(corners_right)

# # # Calibrate left and right cameras
# # ret_left, cameraMatrix_left, distCoeffs_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
# #     object_points_list, image_points_left_list, imgL.shape[::-1], None, None)
# # ret_right, cameraMatrix_right, distCoeffs_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
# #     object_points_list, image_points_right_list, imgR.shape[::-1], None, None)

# # # Calibrate stereo cameras
# # retval, cameraMatrix_left, distCoeffs_left, cameraMatrix_right, distCoeffs_right, R, T, E, F = cv2.stereoCalibrate(
# #     object_points_list, image_points_left_list, image_points_right_list, cameraMatrix_left, distCoeffs_left, cameraMatrix_right, distCoeffs_right, imgL.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC)

# # R, T = cv2.stereoCalibrate(objectPoints, image_points_left_list, image_points_right_list, cameraMatrix_left, distCoeffs_left, cameraMatrix_right, distCoeffs_right, imgL.shape[1] , flags=cv2.CALIB_FIX_INTRINSIC)
# # R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
# #     cameraMatrix_left, distCoeffs_left, cameraMatrix_right, distCoeffs_right,
# #     (imgL.shape[1], imgL.shape[0]), R, T, alpha=0)

# # # Apply rectification to both images
# # rect_left = cv2.remap(imgL, R1, P1, cv2.INTER_LINEAR)
# # rect_right = cv2.remap(imgR, R2, P2, cv2.INTER_LINEAR)

# # Detect and match features
# # TODO: Choose your own feature detection and matching algorithm, and tune its parameters if necessary
# sift = cv2.SIFT_create()
# kp1, des1 = sift.detectAndCompute(imgL, None)
# kp2, des2 = sift.detectAndCompute(imgR, None)
# bf = cv2.BFMatcher()
# matches = bf.knnMatch(des1, des2, k=2)
# plt.imshow(matches,'gray')

# # Apply ratio test to remove false matches
# good_matches = []
# for m,n in matches:
#     if m.distance < 0.75 * n.distance:
#         good_matches.append(m)

# # Compute disparity map
# # TODO: Choose your own disparity computation method, and tune its parameters if necessary
# disparity = np.zeros_like(rect_left)
# for match in good_matches:
#     x_left, y_left = kp1[match.queryIdx].pt
#     x_right, y_right = kp2[match.trainIdx].pt
#     disparity[int(y_left), int(x_left)] = abs(x_left - x_right)

# # Refine disparity map
# # TODO: Choose your own disparity refinement method, and tune its parameters if necessary
# disparity = cv2.medianBlur(disparity, ksize=5)

# # Generate depth map
# # TODO: Define your own camera parameters and stereo baseline, and compute depth map accordingly
# focal_length = 0.8 * image_width  # Focal length in pixels
# baseline = 0.1  # Stereo baseline in meters
# depth_map = np.zeros_like(disparity, dtype=np.float32)
# for y in range(depth_map.shape[0]):
#     for x in range(depth_map.shape[1]):
#         if disparity[y, x] > 0:
#             depth_map[y, x] = (focal_length * baseline) / disparity[y, x]

# # Visualize depth map
# cv2.imshow("Depth Map", depth_map)
# cv2.waitKey(0)
#cv2.destroyAllWindows()


# import numpy as np
# import cv2

# # Define the size of the chessboard pattern
# pattern_size = (9, 6)

# # Define the object points of the pattern (assuming the pattern is fixed on a flat surface)
# object_points = np.zeros((np.prod(pattern_size), 3), dtype=np.float32)
# object_points[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# # Initialize arrays to store object points and image points from all images
# object_points_list = []
# image_points_left_list = []
# image_points_right_list = []

# # Load calibration images for both left and right cameras
# for i in range(num_images):
#     left_img = cv2.imread("left_calib_{}.png".format(i))
#     right_img = cv2.imread("right_calib_{}.png".format(i))

#     # Convert images to grayscale
#     gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
#     gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

#     # Find chessboard corners in the images
#     ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size)
#     ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size)

#     # If corners are found in both images, add object points and image points to the lists
#     if ret_left and ret_right:
#         object_points_list.append(object_points)
#         image_points_left_list.append(corners_left)
#         image_points_right_list.append(corners_right)

# # Calibrate left and right cameras
# ret_left, cameraMatrix_left, distCoeffs_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
#     object_points_list, image_points_left_list, gray_left.shape[::-1], None, None)
# ret_right, cameraMatrix_right, distCoeffs_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
#     object_points_list, image_points_right_list, gray_right.shape[::-1], None, None)

# # Calibrate stereo cameras
# retval, cameraMatrix_left, distCoeffs_left, cameraMatrix_right, distCoeffs_right, R, T, E, F = cv2.stereoCalibrate(
#     object_points_list, image_points_left_list, image_points_right_list, cameraMatrix_left, distCoeffs_left, cameraMatrix_right, distCoeffs_right, gray_left.shape[::-1], flags=cv2.CALIB_FIX_INTRINSIC)

# # Compute rectification maps for both cameras
# R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
#     cameraMatrix_left, distCoeffs_left, cameraMatrix_right, distCoeffs_right,
#     gray_left.shape[::-1], R, T, alpha=0)

# # Save the rectification maps for later use
# np.save


# import numpy as np
# import cv2

# # TODO: Load stereo pair images
# left_img = cv2.imread("left_image.png")
# right_img = cv2.imread("right_image.png")

# # TODO: Resize images if necessary
# img_size = left_img.shape[:2][::-1]  # (width, height)
# if img_size[0] > 640:  # resize if width > 640
#     scale = 640 / img_size[0]
#     img_size = (640, int(img_size[1] * scale))
#     left_img = cv2.resize(left_img, img_size)
#     right_img = cv2.resize(right_img, img_size)

# # TODO: Obtain camera intrinsic and extrinsic parameters for left and right cameras, 
# # as well as stereo calibration results from this camera https://a.aliexpress.com/_EIVKwsf 
# # (assuming calibration results are saved in "stereo_calib.npz")
# calib_data = np.load("stereo_calib.npz")
# cameraMatrix_left = calib_data["cameraMatrix_left"]
# distCoeffs_left = calib_data["distCoeffs_left"]
# cameraMatrix_right = calib_data["cameraMatrix_right"]
# distCoeffs_right = calib_data["distCoeffs_right"]
# R = calib_data["R"]
# T = calib_data["T"]
# img_size = tuple(calib_data["img_size"])
# R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cameraMatrix_left, distCoeffs_left,
#                                                     cameraMatrix_right, distCoeffs_right,
#                                                     img_size, R, T)
# map1_left, map2_left = cv2.initUndistortRectifyMap(cameraMatrix_left, distCoeffs_left, R1, P1, img_size, cv2.CV_32FC1)
# map1_right, map2_right = cv2.initUndistortRectifyMap(cameraMatrix_right, distCoeffs_right, R2, P2, img_size, cv2.CV_32FC1)

# # Rectify images
# left_rectified = cv2.remap(left_img, map1_left, map2_left, cv2.INTER_LINEAR)
# right_rectified = cv2.remap(right_img, map1_right, map2_right, cv2.INTER_LINEAR)

# # TODO: Convert images to grayscale
# left_gray = cv2.cvtColor(left_rectified, cv2.COLOR_BGR2GRAY)
# right_gray = cv2.cvtColor(right_rectified, cv2.COLOR_BGR2GRAY)

# # TODO: Compute disparity map using StereoBM
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
# disparity = stereo.compute(left_gray, right_gray)

# # TODO: Normalize and display the disparity map
# disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
# cv2.imshow("Disparity", disparity_normalized)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
