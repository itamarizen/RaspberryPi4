import cv2
import numpy as np

# Load the stereo images
imgL = cv2.imread(r'C:\Users\User\RaspberryPi4_PCfolder\flowerL.jpeg',cv2.IMREAD_GRAYSCALE)#.astype(np.uint8)
imgR = cv2.imread(r'C:\Users\User\RaspberryPi4_PCfolder\flowerR.jpeg',cv2.IMREAD_GRAYSCALE)#.astype(np.uint8)
imgL = cv2.resize(imgL, (imgR.shape[1], imgR.shape[0]))


# Check if the images have the same dimensions
if imgL.shape != imgR.shape:
    raise ValueError('The left and right images have different dimensions')

# Define the intrinsic camera parameters
focal_length = 3740.0  # example value, adjust as needed
cx, cy = imgL.shape[1] / 2, imgL.shape[0] / 2
K = np.array([[focal_length, 0, cx],
              [0, focal_length, cy],
              [0, 0, 1]])

# Define the stereo camera parameters
baseline = 174.0  # example value, adjust as needed
T = np.array([[-baseline, 0, 0]])

# Compute the rectification matrices and disparity-to-depth mapping matrix
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, np.zeros(5), K, np.zeros(5), imgL.shape[::-1], R=np.eye(3), T=T.transpose(), flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
mapLx, mapLy = cv2.initUndistortRectifyMap(K, np.zeros(5), R1, P1, imgL.shape[::-1], cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K, np.zeros(5), R2, P2, imgR.shape[::-1], cv2.CV_32FC1)

# Rectify the stereo images
imgL_rect = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
imgR_rect = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

# Set the window size and maximum disparity for stereo matching
window_size = 10
max_disparity = 30

# Compute the disparity map using the SAD algorithm
disparity_SAD = np.zeros_like(imgL_rect)
for disparity in range(max_disparity):
    # Shift the right image by the current disparity and compute the absolute difference
    if disparity == 0:
        diff = cv2.absdiff(imgL_rect, imgR_rect)
    else:
        imgR_rect_shifted = np.roll(imgR_rect, -disparity, axis=1)
        diff = cv2.absdiff(imgL_rect[:, :-disparity], imgR_rect_shifted[:, :-disparity])

    # Compute the sum of absolute differences in the window around each pixel
    mask = np.ones((window_size, window_size), dtype=np.uint8) * 255
    
    # Compute the sum of absolute differences in the window around each pixel
    SAD = cv2.matchTemplate(diff, mask, cv2.TM_SQDIFF_NORMED)


    # Resize the disparity map to match the size of the SAD map
    disparity_resized = cv2.resize(disparity_SAD[:, :SAD.shape[1]], SAD.shape[::-1])
    disparity_SAD_broadcast = np.broadcast_to(disparity_resized, SAD.shape)

# Normalize the disparity map
disparity_norm = cv2.normalize(disparity_SAD, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Compute the depth map using triangulation
depth_map = np.zeros_like(imgL_rect, np.float32)
for y in range(depth_map.shape[0]):
    for x in range(depth_map.shape[1]):
        if disparity_norm[y, x] == 0:
            depth_map[y, x] = 0
        else:
            depth_map[y, x] = (focal_length * baseline) / disparity_norm[y, x]

# Scale the depth map to display as an image
depth_map_scaled = cv2.normalize(depth_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# Display the rectified images and depth map
cv2.imshow('Rectified Image Left', imgL_rect)
cv2.imshow('Rectified Image Right', imgR_rect)
cv2.imshow('Depth Map', depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()
# import cv2
# import numpy as np

# # Load stereo images
# imgL = cv2.imread(r'C:\Users\User\RaspberryPi4_PCfolder\flowerL.jpeg',cv2.IMREAD_GRAYSCALE).astype(np.uint8)
# imgR = cv2.imread(r'C:\Users\User\RaspberryPi4_PCfolder\flowerR.jpeg',cv2.IMREAD_GRAYSCALE).astype(np.uint8)
# imgL = cv2.resize(imgL,(455,455), interpolation = cv2.INTER_AREA)
# imgR = cv2.resize(imgR,(455,455), interpolation = cv2.INTER_AREA)

# # Set the window size and maximum disparity for stereo matching
# window_size = 3
# max_disparity = 16

# # Create multiple stereoSGBM objects with different parameter settings
# stereo1 = cv2.StereoSGBM_create(minDisparity=0,
#                                 numDisparities=max_disparity,
#                                 blockSize=window_size,
#                                 P1=8*3*window_size**2,
#                                 P2=32*3*window_size**2)
# stereo2 = cv2.StereoSGBM_create(minDisparity=0,
#                                 numDisparities=max_disparity,
#                                 blockSize=window_size,
#                                 P1=8*2*window_size**2,
#                                 P2=32*2*window_size**2)
# stereo3 = cv2.StereoSGBM_create(minDisparity=0,
#                                 numDisparities=max_disparity,
#                                 blockSize=window_size,
#                                 P1=8*4*window_size**2,
#                                 P2=32*4*window_size**2)

# # Compute the disparity map using each stereo algorithm
# disparity1 = stereo1.compute(imgL, imgR)
# disparity2 = stereo2.compute(imgL, imgR)
# disparity3 = stereo3.compute(imgL, imgR)

# # Combine the disparity maps using weighted averaging
# disparity = (disparity1 / 4.0) + (disparity2 / 2.0) + (disparity3 / 4.0)

# # Normalize the disparity map to the range [0, 255]
# disparity_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)

# # Convert the disparity map to a depth map
# focal_length = 10  # example value, adjust as needed
# baseline = 10  # example value, adjust as needed
# depth_map = (focal_length * baseline) / (disparity_norm + 1e-6)

# # Apply object labeling to the depth map
# ret, thresh = cv2.threshold(depth_map, 0, 255, cv2.THRESH_BINARY)
# # Convert the input image to grayscale if needed
# if len(thresh.shape) > 2:
#     thresh_gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
# else:
#     thresh_gray = thresh

# # Ensure that the input image is in the correct format
# if thresh_gray.dtype != np.uint8:
#     thresh_gray = thresh_gray.astype(np.uint8)

# # Apply object labeling to the depth map
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for i, c in enumerate(contours):
#     area = cv2.contourArea(c)
#     if area > 50:
#         cv2.drawContours(depth_map, contours, i, (255, 255, 255), -1)

# # Apply blob detection to the depth map
# params = cv2.SimpleBlobDetector_Params()
# params.filterByArea = True
# params.minArea = 50
# params.filterByCircularity = False
# params.filterByConvexity = False
# params.filterByInertia = False
# detector = cv2.SimpleBlobDetector_create(params)
# keypoints = detector.detect(depth_map.astype(np.uint8))
# im_with_keypoints = cv2.drawKeypoints(depth_map.astype(np.uint8), keypoints, np.array([]), (0, 255, 0),
#                                       cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# # Display the depth map with labeled objects and detected blobs
# cv2.imshow('Depth Map', im_with_keypoints)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
