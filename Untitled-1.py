import cv2
import numpy as np

# Load depth map
depth_map = cv2.imread('bike.jpg',0)

# Perform thresholding
_,thresh = cv2.threshold(depth_map,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Find contours
contours,_ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# Initialize list to store obstacles
obstacles = []

# Iterate over each contour
for contour in contours:
    # Calculate area of contour
    area = cv2.contourArea(contour)
    # Filter out contours with area greater than threshold
    if area > 20:
        obstacles.append(contour)

# Draw obstacles on image
cv2.drawContours(depth_map,obstacles,-1,(0,0,255),3)

# Display the output
cv2.imshow(depth_map,'gray')
cv2.waitKey(0)