import cv2
import pygame
pygame.init()
import numpy as np
#import pdb;pdb.set_trace()

# Read image file
img = cv2.imread("bike.jpg",0) # remove 0 to get colored image

# Create a named window with the WINDOW_AUTOSIZE flag
cv2.namedWindow("Image", cv2.WINDOW_NORMAL) #cv2.WINDOW_AUTOSIZE
# Resize the window to match the image size
cv2.resizeWindow("Image", img.shape[1], img.shape[0])

# Check if the image was successfully loaded
if img is not None:
    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
else:
    print("Failed to load image")

# Detect edges using the Canny algorithm
edges = cv2.Canny(img, 50, 150)
# Create a named window with the WINDOW_AUTOSIZE flag
cv2.namedWindow("edges", cv2.WINDOW_NORMAL) #cv2.WINDOW_AUTOSIZE
# Resize the window to match the image size
cv2.resizeWindow("edges", img.shape[1], img.shape[0])
cv2.imshow("edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Apply a morphological transformation to fill in gaps and eliminate noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
# Create a named window with the WINDOW_AUTOSIZE flag
cv2.namedWindow("kernel", cv2.WINDOW_NORMAL) #cv2.WINDOW_AUTOSIZE
# Resize the window to match the image size
cv2.resizeWindow("kernel", img.shape[1], img.shape[0])
cv2.imshow("kernel", kernel)
cv2.waitKey(0)
cv2.destroyAllWindows()

closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
# Create a named window with the WINDOW_AUTOSIZE flag
cv2.namedWindow("closed", cv2.WINDOW_NORMAL) #cv2.WINDOW_AUTOSIZE
# Resize the window to match the image size
cv2.resizeWindow("closed", img.shape[1], img.shape[0])
cv2.imshow("closed", closed)
cv2.waitKey(0)

if cv2.countNonZero(closed) > 0:
    # Play a warning sound using pygame
    pygame.mixer.music.load("warning.wav")
    pygame.mixer.music.play()

    # compute the difference between the current frame and the previous frame
    frame_diff = cv2.absdiff(frame0, frame1)
    # threshold the difference to create a binary motion mask
    motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
    cv2.connectedComponents()
    cv2.SimpleBlobDetector()
    cv2.HoughLines()


cv2.destroyAllWindows()
