# `import cv2
# import numpy as np
# from tkinter import *
# from tkinter import filedialog
# import time
# prev_distance = 0

# # Define GUI for user choice of preprocessing
# root = Tk()
# root.title("Preprocessing Options")

# def preprocess():
#     global option
#     option = variable.get()
#     root.destroy()

# Label(root, text="Choose a preprocessing option:").pack()
# variable = StringVar(root, "None")
# Radiobutton(root, text="None", variable=variable, value="None").pack(anchor=W)
# Radiobutton(root, text="Gaussian Blur", variable=variable, value="Gaussian Blur").pack(anchor=W)
# Radiobutton(root, text="Median Blur", variable=variable, value="Median Blur").pack(anchor=W)
# Radiobutton(root, text="Bilateral Filter", variable=variable, value="Bilateral Filter").pack(anchor=W)
# Radiobutton(root, text="Gaussian Noise", variable=variable, value="Gaussian Noise").pack(anchor=W)
# Button(root, text="OK", command=preprocess).pack()
# root.mainloop()

# # Load stereo images
# root = Tk()
# root.filename1 = filedialog.askopenfilename(title="Select first stereo image")
# root.filename2 = filedialog.askopenfilename(title="Select second stereo image")
# root.destroy()

# imgL = cv2.imread(root.filename1)
# imgR = cv2.imread(root.filename2)

# # Preprocess stereo images
# if option == "Gaussian Blur":
#     imgL = cv2.GaussianBlur(imgL, (5, 5), 0)
#     imgR = cv2.GaussianBlur(imgR, (5, 5), 0)
# elif option == "Median Blur":
#     imgL = cv2.medianBlur(imgL, 5)
#     imgR = cv2.medianBlur(imgR, 5)
# elif option == "Bilateral Filter":
#     imgL = cv2.bilateralFilter(imgL, 9, 75, 75)
#     imgR = cv2.bilateralFilter(imgR, 9, 75, 75)
# elif option == "Gaussian Noise":
#     noise = np.zeros(imgL.shape, np.uint8)
#     cv2.randn(noise, 0, 50)
#     imgL = cv2.add(imgL, noise)
#     imgR = cv2.add(imgR, noise)

# # Create stereo matcher object and perform depth mapping
# stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
# disparity = stereo.compute(cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY))

# # Normalize and convert disparity map to 8-bit image
# min_disp = disparity.min()
# max_disp = disparity.max()
# disparity = np.uint8(255 * (disparity - min_disp) / (max_disp - min_disp))

# # Perform object detection and motion tracking
# object_detected = False
# object_warned = False
# while True:
#     # Capture current frame from stereo camera
#     imgL = cv2.imread(root.filename1)
#     imgR = cv2.imread(root.filename2)
    
#     # Preprocess stereo images
#     if option == "Gaussian Blur":
#         imgL = cv2.GaussianBlur(imgL, (5, 5), 0)
#         imgR = cv2.GaussianBlur(imgR, (5, 5), 0)
#     elif option == "Median Blur":
#         imgL = cv2.medianBlur(imgL, 5)
#         imgR = cv2.medianBlur(imgR, 5)
#     elif option == "Bilateral Filter":
#         imgL = cv2.bilateralFilter(imgL, 9, 75, 75)
#         imgR = cv2.bilateralFilter(imgR, 9, 75, 75)
#     elif option == "Gaussian Noise":
#         noise = np.zeros(imgL.shape, np.uint8)
#         cv2.randn(noise, 0, 50)
#         imgL = cv2.add(imgL, noise)
#         imgR = cv2.add(imgR, noise)

#     # Perform depth mapping
#     disparity = stereo.compute(cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY), cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY))
#     min_disp = disparity.min()
#     max_disp = disparity.max()
#     disparity = np.uint8(255 * (disparity - min_disp) / (max_disp - min_disp))
    
#     # Perform object detection and motion tracking
#     img = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
#     threshold = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
#     contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     for c in contours:
#         # Calculate object distance and movement within two shots
#         area = cv2.contourArea(c)
#         if area > 100:
#             object_detected = True
#             object_warned = False
#             M = cv2.moments(c)
#             cx = int(M["m10"] / M["m00"])
#             cy = int(M["m01"] / M["m00"])
#             distance = disparity[cy, cx]
#             movement = abs(distance - prev_distance)
#             prev_distance = distance
            
#             # Warn user if object is too close
#             if distance < 100 and not object_warned:
#                 print("Object too close!")
#                 object_warned = True
            
#             # Draw bounding box and label for object
#             (x, y, w, h) = cv2.boundingRect(c)
#             cv2.rectangle(imgL, (x, y), (x + w, y + h), (0, 255, 0), 2)
#             cv2.putText(imgL, "Object {}cm".format(distance), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
#     # Display output images to the screen
#     cv2.imshow("Left Camera", imgL)
#     cv2.imshow("Right Camera", imgR)
    
#     # Store current frame data for motion tracking
#     prev_frame = img
    
#     # Exit loop if user presses 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cv2.destroyAllWindows()

import cv2
import numpy as np
import time
import os
from tkinter import *
from tkinter import filedialog
from PIL import Image 
from PIL import ImageTk 

# function to preprocess stereo images
def preprocess():
    global processed_left, processed_right, Q

    # convert images to grayscale
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # find corresponding points in stereo images
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(left_gray, right_gray)

    # rectify and align images
    h, w = left_gray.shape
    _, _, _, _, _, Q = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, (w, h), R, T)
    map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, (w, h), cv2.CV_32FC1)
    processed_left = cv2.remap(left, map1, map2, cv2.INTER_LINEAR)
    map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, (w, h), cv2.CV_32FC1)
    processed_right = cv2.remap(right, map1, map2, cv2.INTER_LINEAR)

# function to detect objects and motion
def detect():
    global processed_left, processed_right, previous_x, too_close, warned
    # convert images to grayscale
    gray_left = cv2.cvtColor(processed_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(processed_right, cv2.COLOR_BGR2GRAY)
    # perform image processing based on user-selected options
    for option in options:
        if option == 'Blur':
            gray_left = cv2.GaussianBlur(gray_left, (5, 5), 0)
            gray_right = cv2.GaussianBlur(gray_right, (5, 5), 0)
        elif option == 'Threshold':
            _, gray_left = cv2.threshold(gray_left, 127, 255, cv2.THRESH_BINARY)
            _, gray_right = cv2.threshold(gray_right, 127, 255, cv2.THRESH_BINARY)
        elif option == 'Canny Edge Detection':
            gray_left = cv2.Canny(gray_left, 100, 200)
            gray_right = cv2.Canny(gray_right, 100, 200)
        elif option == 'Dilation':
            kernel = np.ones((5, 5), np.uint8)
            gray_left = cv2.dilate(gray_left, kernel, iterations=1)
            gray_right = cv2.dilate(gray_right, kernel, iterations=1)
        elif option == 'Erosion':
            kernel = np.ones((5, 5), np.uint8)
            gray_left = cv2.erode(gray_left, kernel, iterations=1)
            gray_right = cv2.erode(gray_right, kernel, iterations=1)
    # find keypoints and descriptors in stereo images
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_left, None)
    kp2, des2 = sift.detectAndCompute(gray_right, None)
    # match keypoints between stereo images
    matcher = cv2.BFMatcher()
    matches = matcher.match(des1, des2)
    # filter out poor matches based on distance
    matches = [m for m in matches if m.distance < 100]
    # draw matches on images
    img_matches = cv2.drawMatches(processed_left, kp1, processed_right, kp2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # find motion between stereo images
    flow = cv2.calcOpticalFlowFarneback(gray_left, gray_right, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    # calculate distance and movement of objects
    objects = []
    for match in matches:
        x1, y1 = kp1[match.queryIdx].pt
        x2, y2 = kp2[match.trainIdx].pt
        depth = Q[3][2] / (Q[2][3] * (x1 - x2) + Q[3][3])
        objects.append((x1, y1, depth, flow[int(y1), int(x1)][0], flow[int(y1), int(x1)][1]))

    # detect too-close objects and play warning sound
    too_close = False
    for obj in objects:
        if obj[2] < min_distance:
            too_close = True
        if not warned:
            os.system("afplay warning.mp3&")
        warned = True
        break
    if not too_close:
        warned = False
    # draw rectangles around objects
    for obj in objects:
        x, y = int(obj[0]), int(obj[1])
        w, h = int(obj[2] * 0.1), int(obj[2] * 0.1)
        cv2.rectangle(processed_left, (x-w, y-h), (x+w, y+h), (0, 255, 0), 2)

    # display images with detections and motion
    cv2.imshow('Stereo Images', np.hstack((img_matches, cv2.cvtColor(flow, cv2.COLOR_BGR2RGB))))
    cv2.imshow('Processed Left Image', processed_left)
    cv2.imshow('Processed Right Image', processed_right)

    # wait for key press
    key = cv2.waitKey(1)
    # exit if escape key is pressed
    if key == 27:
        cv2.destroyAllWindows()
        exit()

# initialize GUI
root = Tk()
root.title("Stereo Vision GUI")

# initialize variables
max_image_width = 100
max_image_height = 100
left = None
right = None
processed_left = None
processed_right = None
cameraMatrix1 = None
cameraMatrix2 = None
distCoeffs1 = None
distCoeffs2 = None
R = None
T = None
R1 = None
P1 = None
R2 = None
P2 = None
Q = None
previous_x = None
too_close = False
warned = False
min_distance = 50
options = []

#function to select left image file
def select_left_image():
    global left
    file_path = filedialog.askopenfilename()
    if file_path != "":
        left = cv2.imread(file_path)
        preprocess()

# function to select right image file
def select_right_image():
    global right
    file_path = filedialog.askopenfilename()
    if file_path != "":
        right = cv2.imread(file_path)
        preprocess()



# function to select camera parameters file
def select_camera_params():
    global cameraMatrix1, cameraMatrix2, distCoeffs1, distCoeffs2, R, T, R1, P1, R2, P2, Q
    file_path = filedialog.askopenfilename()
    if file_path != "":
        with open(file_path, 'r') as f:
            data = json.load(f)
    cameraMatrix1 = np.array(data['cameraMatrix1'])
    cameraMatrix2 = np.array(data['cameraMatrix2'])
    distCoeffs1 = np.array(data['distCoeffs1'])
    distCoeffs2 = np.array(data['distCoeffs2'])
    R = np.array(data['R'])
    T = np.array(data['T'])
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, left.shape[:2], R, T)

#function to preprocess stereo images
def preprocess():
    global left, right, processed_left, processed_right
    if left is not None and right is not None:
    # rectify and align images
        left = cv2.undistort(left, cameraMatrix1, distCoeffs1)
        right = cv2.undistort(right, cameraMatrix2, distCoeffs2)
        map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, left.shape[:2], cv2.CV_32FC1)
        left = cv2.remap(left, map1, map2, cv2.INTER_LINEAR)
        map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, right.shape[:2], cv2.CV_32FC1)
        right = cv2.remap(right, map1, map2, cv2.INTER_LINEAR)
    # apply selected image processing methods
    for option in options:
        if option == "Gaussian Blur":
            left = cv2.GaussianBlur(left, (5, 5), 0)
            right = cv2.GaussianBlur(right, (5, 5), 0)
        elif option == "Median Blur":
            left = cv2.medianBlur(left, 5)
            right = cv2.medianBlur(right, 5)
        elif option == "Bilateral Filter":
            left = cv2.bilateralFilter(left, 9, 75, 75)
            right = cv2.bilateralFilter(right, 9, 75, 75)
        elif option == "Canny Edge Detection":
            left = cv2.Canny(left, 100, 200)
            right = cv2.Canny(right, 100, 200)
        elif option == "Sobel Edge Detection":
            left = cv2.Sobel(left, cv2.CV_8U, 1, 0, ksize=5)
            right = cv2.Sobel(right, cv2.CV_8U, 1, 0, ksize=5)
    processed_left = left.copy()
    processed_right = right.copy()

# function to update selected options
def update_options():
    global options
    options = [var.get() for var in option_vars]
    preprocess()

# create GUI widgets
left_button = Button(root, text="Select Left Image", command=select_left_image)
left_button.pack()
right_button = Button(root, text="Select Right Image", command=select_right_image)
right_button.pack()
camera_button = Button(root, text="Select Camera Parameters", command=select_camera_params)
camera_button.pack()

Label(root, text="Preprocessing Options:").pack()
option_vars = [StringVar() for _ in range(5)]
option_vars[0].set("Gaussian Blur")
option_vars[1].set("Median Blur")
option_vars[2].set("Bilateral Filter")
option_vars[3].set("Canny Edge Detection")
option_vars[4].set("Sobel Edge Detection")
for i, option in enumerate(["Gaussian Blur", "Median Blur", "Bilateral Filter", "Canny Edge Detection", "Sobel Edge Detection"]):
    Checkbutton(root, text=option, variable=option_vars[i], onvalue=option, offvalue="", command=update_options).pack()

# create canvas for displaying images
canvas = Canvas(root, width=2 * max_image_width + 20, height=max_image_height + 20)
canvas.pack()

def display_images():
    global canvas, left, right, processed_left, processed_right, matches
    canvas.delete("all")
    # display left and right images side by side
    canvas.create_image(10, 10, anchor=NW, image=ImageTk.PhotoImage(Image.fromarray(left)))
    canvas.create_image(max_image_width + 20, 10, anchor=NW, image=ImageTk.PhotoImage(Image.fromarray(right)))
    # display processed left and right images side by side
    canvas.create_image(10, max_image_height + 20, anchor=NW, image=ImageTk.PhotoImage(Image.fromarray(processed_left)))
    canvas.create_image(max_image_width + 20, max_image_height + 20, anchor=NW, image=ImageTk.PhotoImage(Image.fromarray(processed_right)))
    # display matches on processed left and right images
    if matches is not None:
        for match in matches:
            x1, y1 = match[0]
            x2, y2 = match[1]
    canvas.create_line(x1+10, y1+max_image_height+20, x2+max_image_width+20, y2+max_image_height+20, fill="red")

root.after(100, display_images)
# start image display loop
display_images()

root.mainloop()