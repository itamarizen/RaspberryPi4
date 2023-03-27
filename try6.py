# import required libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt

A = 1
def show_image(image, windoe_name):
    # Create a named window with the WINDOW_AUTOSIZE flag
    cv2.namedWindow(windoe_name, cv2.WINDOW_NORMAL) #cv2.WINDOW_AUTOSIZE
    # Resize the window to match the image size
    cv2.resizeWindow(windoe_name, image.shape[1], image.shape[0])
    # Display the combined image
    cv2.imshow(windoe_name, image)
    cv2.waitKey(0)

def preproceesing(imgL, imgR, choice):
    # Apply selected deep mapping algorithm
    if choice == '1': # Image smoothing
        imgL = cv2.GaussianBlur(imgL, (5, 5), 0)
        imgR = cv2.GaussianBlur(imgR, (5, 5), 0)
        output = cv2.hconcat([imgL, imgR])
        output = cv2.resize(output, (640, 240))
    elif choice == '2': # Image thresholding
        ret, imgL = cv2.threshold(imgL, 127, 255, cv2.THRESH_BINARY)
        ret, imgR = cv2.threshold(imgR, 127, 255, cv2.THRESH_BINARY)
        output = cv2.hconcat([imgL, imgR])
        output = cv2.resize(output, (640, 240))
    elif choice == '3': # Blob detection
        detector = cv2.SimpleBlobDetector_create()
        keypointsL = detector.detect(imgL)
        keypointsR = detector.detect(imgR)
        imgL = cv2.drawKeypoints(imgL, keypointsL, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        imgR = cv2.drawKeypoints(imgR, keypointsR, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        output = cv2.hconcat([imgL, imgR])
        output = cv2.resize(output, (640, 240))
    elif choice == '4': # Edge detection
        imgL = cv2.Canny(imgL, 100, 200)
        imgR = cv2.Canny(imgR, 100, 200)
        output = cv2.hconcat([imgL, imgR])
        output = cv2.resize(output, (640, 240))
    elif choice == '5': # Image sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        imgL = cv2.filter2D(imgL, -1, kernel)
        imgR = cv2.filter2D(imgR, -1, kernel)
        output = cv2.hconcat([imgL, imgR])
        output = cv2.resize(output, (640, 240))
    else:
        print("Invalid choice, using default (smoothing)...")
        imgL = cv2.GaussianBlur(imgL, (5, 5), 0)
        imgR = cv2.GaussianBlur(imgR, (5, 5), 0)
        output = cv2.hconcat([imgL, imgR])
        output = cv2.resize(output, (640, 240))
    return output

def stereo(imgL, imgR, choice):
    # Apply selected deep mapping algorithm
    if choice == '1': # stereo_bm
        output = stereo_bm(imgL, imgR)
    elif choice == '2': # stereo_bm_GaussianBlur
        output = stereo_bm_GaussianBlur(imgL, imgR)
    elif choice == '3': # stereo_sgbm_wls
        output = stereo_sgbm_wls(imgL, imgR)
    elif choice == '4': # stereo_bm_noise_removal
        output = stereo_bm_noise_removal(imgL, imgR)
    else:
        print("Invalid choice, using default (stereo_bm)...")
        output = stereo_bm(imgL, imgR)
    return output

def create_3d_mapping_image(imgL, imgR):

    # Find the key points and descriptors in both images using SIFT algorithm
    #sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.ORB_create()
    kp1, des1 = sift.detectAndCompute(imgL, None)
    kp2, des2 = sift.detectAndCompute(imgR, None)

    # Match the descriptors in both images
    matcher = cv2.BFMatcher()
    matches = matcher.match(des1, des2)

    # Calculate the homography between the two images
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Use the homography to warp one image onto the other
    h, w = imgL.shape
    warped = cv2.warpPerspective(imgL, M, (w, h))

    # Blend the two images to create the 3D mapping output image
    alpha = 0.5
    beta = 1.0 - alpha
    output = cv2.addWeighted(imgR, alpha, warped, beta, 0.0)

    return output


def stereo_bm_GaussianBlur(imgL, imgR, numDisparities=16, blockSize=15):

    # Compute the disparity map using the block matching algorithm
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparity = stereo.compute(imgL, imgR)

    # Normalize the disparity map for display purposes
    normalized_disparity = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Apply a Gaussian blur to the normalized disparity map to reduce noise
    blurred_disparity = cv2.GaussianBlur(normalized_disparity, (5, 5), 0)

    # Return the enhanced depth mapping image
    return blurred_disparity


def stereo_bm(imgL, imgR, numDisparities=16, blockSize=15):
    # Compute the disparity map using the block matching algorithm
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparity = stereo.compute(imgL, imgR)

    # Return the disparity map
    return disparity


def stereo_sgbm_wls(imgL, imgR, numDisparities=16, blockSize=15, lmbda=80000, sigma=1.2):

    # Compute the disparity map using the block matching algorithm
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=numDisparities, blockSize=blockSize)
    disparity = stereo.compute(imgL, imgR)

    # Apply a weighted least-squares filter to the disparity map
    filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
    filter.setLambda(lmbda)
    filter.setSigmaColor(sigma)
    filtered_disparity = filter.filter(disparity, imgL, None, imgR)

    # Return the filtered disparity map
    return filtered_disparity


def stereo_bm_noise_removal(imgL, imgR, numDisparities=16, blockSize=15, filterType="median"):

    # Compute the disparity map using the block matching algorithm
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)
    disparity = stereo.compute(imgL, imgR)

    # Remove noise from the disparity map using a filter
    if filterType == "median":
        filtered_disparity = cv2.medianBlur(disparity, 3)
    elif filterType == "bilateral":
        filtered_disparity = cv2.bilateralFilter(disparity, 5, 75, 75)
    else:
        filtered_disparity = disparity

    # Return the filtered disparity map
    return filtered_disparity


if (A==1):
    # read two input images as grayscale images
    imgL = cv2.imread('L.png',0)
    imgR = cv2.imread('R.png',0)
    # show_image(imgL,'L.png')
    # show_image(imgR,'R.png')

    # Initiate and StereoBM object
    stereo = cv2.StereoSGBM_create(numDisparities=16, blockSize=25)

    # compute the disparity map
    disparity = stereo.compute(imgL,imgR)
    plt.subplot(131), plt.imshow(imgL, cmap = "gray"),plt.axis('off')
    plt.subplot(132), plt.imshow(imgR, cmap = "gray"),plt.axis('off')
    plt.subplot(133), plt.imshow(disparity, cmap = "gray"),plt.axis('off')
    plt.imshow(disparity,'gray')
    plt.show()
    disparity.shape

if (A==2):
    # read two input images
    imgL = cv2.imread('aloeL.jpg',0)
    imgR = cv2.imread('aloeR.jpg',0)
    show_image(imgL,'aloeL.jpg')
    show_image(imgR,'aloeR.jpg')

    # Initiate and StereoBM object
    stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)

    # compute the disparity map
    disparity = stereo.compute(imgL,imgR)
    disparity1 = stereo.compute(imgR,imgL)
    plt.subplot(131), plt.imshow(imgL, cmap = "gray"),plt.axis('off')
    plt.subplot(132), plt.imshow(imgR, cmap = "gray"),plt.axis('off')
    plt.subplot(133), plt.imshow(disparity, cmap = "gray"),plt.axis('off')
    plt.show()