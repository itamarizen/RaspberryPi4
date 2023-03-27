import cv2
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
now = datetime.now()

# Load the stereo images and resize the right image to match the left image size
imgL = cv2.imread(r'C:\Users\User\RaspberryPi4_PCfolder\flowerL.jpeg', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread(r'C:\Users\User\RaspberryPi4_PCfolder\flowerR.jpeg', cv2.IMREAD_GRAYSCALE)
imgR = cv2.resize(imgR, (imgL.shape[1], imgL.shape[0]))

# Define the intrinsic camera parameters
focal_length = 50  # example value, adjust as needed
cx, cy = imgL.shape[1] / 2, imgL.shape[0] / 2
K = np.array([[focal_length, 0, cx],
              [0, focal_length, cy],
              [0, 0, 1]])

# Define the stereo camera parameters
baseline = 0.1  # example value, adjust as needed
T = np.array([[-baseline, 0, 0]])

# Compute the rectification matrices and disparity-to-depth mapping matrix
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, np.zeros(5), K, np.zeros(5), imgL.shape[::-1], R=np.eye(3), T=T.transpose(), flags=cv2.CALIB_ZERO_DISPARITY, alpha=-1)
mapLx, mapLy = cv2.initUndistortRectifyMap(K, np.zeros(5), R1, P1, imgL.shape[::-1], cv2.CV_32FC1)
mapRx, mapRy = cv2.initUndistortRectifyMap(K, np.zeros(5), R2, P2, imgR.shape[::-1], cv2.CV_32FC1)

# Rectify the stereo images
imgL = cv2.remap(imgL, mapLx, mapLy, cv2.INTER_LINEAR)
imgR = cv2.remap(imgR, mapRx, mapRy, cv2.INTER_LINEAR)

# Try different values of numDisparities and blockSize
for num_disp in [16, 32, 64, 128]:
    for block_size in [3, 5, 7]:
        # Compute the disparity map using the SGBM algorithm
        stereo = cv2.StereoSGBM_create(numDisparities=num_disp, blockSize=block_size)
        disparity = stereo.compute(imgL, imgR)

        # Convert the disparity map to a depth map using the disparity-to-depth mapping matrix
        depth = cv2.reprojectImageTo3D(disparity, Q)

        # Compute the average depth of the flower region (you can adjust this to your specific use case)
        avg_depth = np.mean(depth[200:400, 200:400, 2])

        disp_norm = cv2.normalize(disparity.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
        depth_norm = cv2.normalize(depth[:, :, 2], None, 0, 255, cv2.NORM_MINMAX)

        # Create a composite image
        imgL_disp = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
        imgR_disp = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)
        disp_norm_bgr = cv2.cvtColor(disp_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        depth_norm_bgr = cv2.cvtColor(depth_norm.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        composite = np.hstack((imgL_disp, imgR_disp, disp_norm_bgr, depth_norm_bgr))

        # Display the results
        plt.imshow(composite)
        plt.show()

        # Save the composite image
        image_name = "composite_{block_size:}_{num_disp}_at_{time}.jpg".format(block_size=str(block_size),num_disp=str(num_disp),time=now.strftime("%H-%M-%S"))
        cv2.imwrite(image_name, composite)

# Replace NaN and Inf values with a default value (e.g. 0)
disparity[np.isnan(disparity)] = 0
disparity[np.isinf(disparity)] = 0

# Display the results
disp_norm = cv2.normalize(disparity.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX)
depth_norm = cv2.normalize(depth[:, :, 2], None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('Disparity Map', disp_norm.astype(np.uint8))
cv2.imshow('Depth Map', depth_norm.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()