# imports
import os,sys
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from matplotlib import pyplot as plt

# add base directory to sys.path
base_dir = r'C:\Users\User\RaspberryPi4_PCfolder\repo_learningopenCV\chapter10'
sys.path.append(os.path.abspath(base_dir))

# ---------------------------- globals ----------------------------- #

# ------------------------------------------------------------------ #
# ---------------------------- functios ---------------------------- #

    
def find_matching_points(imgL, imgR):
    # Convert the images to grayscale
    grayL = imgL
    grayR = imgR
    
    # Create the StereoBM object and set the parameters
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
    
    # Compute the disparity map
    disparity = stereo.compute(grayL, grayR)
    
    # Find the matching points using the disparity map
    matches = []
    for i in range(disparity.shape[0]):
        for j in range(disparity.shape[1]):
            if disparity[i, j] > 0:
                matches.append((j, i))

    show_image(matches,"sx")                
    return matches


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


def object_detection(imgL,imgR):
    # Perform object detection and segmentation using OpenCV's blob detector
    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByConvexity = True
    params.minConvexity = 0.9
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs in the depth map
    keypoints = detector.detect(imgL)

    # Label the blobs with their depth information
    for kp in keypoints:
        # Compute the depth of the blob using the disparity value at the center of the blob
        x, y = int(kp.pt[0]), int(kp.pt[1])
        disparity_value = imgL[y, x]
        depth = 0.2 * 3740 / (disparity_value + 0.0001)  # Formula to compute depth from disparity

        # Label the blob with its depth
        text = "Depth: {:.2f} m".format(depth)
        cv2.putText(imgL, text, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 0, 255), 1)

    # Display the labeled depth map
    show_image(imgL,"Depth map with displacement labels")

    # Compute the displacement of each blob between the two shots
    for kp in keypoints:
        # Get the position of the blob in the previous shot
        x, y = int(kp.pt[0]), int(kp.pt[1])
        disparity_value = imgL[y, x]
        depth = 0.2 * 3740 / (disparity_value + 0.0001)
        prev_pos = np.array([x, y, depth])

        # Get the position of the blob in the current shot
        x_new, y_new = int(kp.pt[0]), int(kp.pt[1])
        disparity_value_new = imgR[y_new, x_new]
        depth_new = 0.2 * 3740 / (disparity_value_new + 0.0001)
        curr_pos = np.array([x_new, y_new, depth_new])

        # Compute the displacement between the two positions
        displacement = np.linalg.norm(curr_pos - prev_pos)

        # Label the blob with its displacement
        text = "Displacement: {:.2f} m".format(displacement)
        cv2.putText(imgR, text, (x_new, y_new), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (255, 0, 255), 1)

    # Display the labeled depth map with displacement information
    show_image(imgR,"Depth map with displacement labels")


def show_image(image, windoe_name):
    # Create a named window with the WINDOW_AUTOSIZE flag
    cv2.namedWindow(windoe_name, cv2.WINDOW_NORMAL) #cv2.WINDOW_AUTOSIZE
    # Resize the window to match the image size
    cv2.resizeWindow(windoe_name, image.shape[1], image.shape[0])
    # Display the combined image
    cv2.imshow(windoe_name, image)
    cv2.waitKey(0) 


def stitch_images_SIFT(imgL, imgR):

    # Detect keypoints and compute descriptors using SIFT
    sift = cv2.xfeatures2d.SIFT_create()
    keypointsL, descriptorsL = sift.detectAndCompute(imgL, None)
    keypointsR, descriptorsR = sift.detectAndCompute(imgR, None)

    # Match keypoints in the two images using a FLANN-based matcher
    matcher = cv2.FlannBasedMatcher()
    matches = matcher.match(descriptorsL, descriptorsR)

    # Sort matches by their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Filter out some of the weaker matches
    num_matches = min(len(matches), 100)
    good_matches = matches[:num_matches]

    # Extract the keypoints for the good matches in each image
    ptsL = np.float32([keypointsL[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ptsR = np.float32([keypointsR[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute the homography matrix that maps points from the right image to the left image
    H, _ = cv2.findHomography(ptsR, ptsL, cv2.RANSAC)

    # Warp the right image using the homography matrix to align it with the left image
    imgR_warped = cv2.warpPerspective(imgR, H, (imgL.shape[1], imgL.shape[0]))

    # Combine the left and warped right images using an alpha blend
    alpha = 0.5
    img_out = cv2.addWeighted(imgL, 1-alpha, imgR_warped, alpha, 0)

    return img_out


def stitch_images_ORB(imgL, imgR):

    # Detect keypoints and compute descriptors using ORB
    orb = cv2.ORB_create()
    keypointsL, descriptorsL = orb.detectAndCompute(imgL, None)
    keypointsR, descriptorsR = orb.detectAndCompute(imgR, None)

    # Match keypoints in the two images using a brute-force matcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptorsL, descriptorsR)

    # Sort matches by their distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Filter out some of the weaker matches
    num_matches = min(len(matches), 100)
    good_matches = matches[:num_matches]

    # Extract the keypoints for the good matches in each image
    ptsL = np.float32([keypointsL[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    ptsR = np.float32([keypointsR[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute the homography matrix that maps points from the right image to the left image
    H, _ = cv2.findHomography(ptsR, ptsL, cv2.RANSAC)

    # Warp the right image using the homography matrix to align it with the left image
    imgR_warped = cv2.warpPerspective(imgR, H, (imgL.shape[1], imgL.shape[0]))

    # Combine the left and warped right images using an alpha blend
    alpha = 0.5
    img_out = cv2.addWeighted(imgL, 1-alpha, imgR_warped, alpha, 0)

    return img_out

def object_detection(image):
    #resize to match the input shape expected by the model
    image = cv2.resize(image, (300, 300))
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    model = cv2.dnn.readNetFromCaffe(
        r'C:\Users\User\RaspberryPi4_PCfolder\RaspberryPi4\objects_data\MobileNetSSD_deploy.prototxt',
        r'C:\Users\User\RaspberryPi4_PCfolder\RaspberryPi4\objects_data\MobileNetSSD_deploy.caffemodel')
    blob_height = 300
    color_scale = 1.0/127.5
    average_color = (127.5, 127.5, 127.5)
    confidence_threshold = 0.5
    labels = ['airplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'dining table', 'dog',
            'horse', 'motorbike', 'person', 'potted plant', 'sheep',
            'sofa', 'train', 'TV or monitor']

    success = True
    while success:

        h, w = image.shape[:2]
        aspect_ratio = w/h

        # Detect objects in the image.

        blob_width = int(blob_height * aspect_ratio)
        blob_size = (blob_width, blob_height)

        blob = cv2.dnn.blobFromImage(
            image, scalefactor=color_scale, size=blob_size,
            mean=average_color)

        model.setInput(blob)
        results = model.forward()

        # Iterate over the detected objects.
        for object in results[0, 0]:
            confidence = object[2]
            if confidence > confidence_threshold:

                # Get the object's coordinates.
                x0, y0, x1, y1 = (object[3:7] * [w, h, w, h]).astype(int)

                # Get the classification result.
                id = int(object[1])
                label = labels[id - 1]

                # Draw a blue rectangle around the object.
                cv2.rectangle(image, (x0, y0), (x1, y1),
                            (255, 0, 0), 2)

                # Draw the classification result and confidence.
                text = '%s (%.1f%%)' % (label, confidence * 100.0)
                cv2.putText(image, text, (x0, y0 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Objects', image)

        k = cv2.waitKey(1)
        if k == 27:  # Escape
            break # was breake

        success = False

# --------------------------------------------------------------------------- #




# ---------------------------- script start here ---------------------------- #
  

# Load stereo images
imgL = cv2.imread(r'C:\Users\User\RaspberryPi4_PCfolder\stereo2-r.jpg',cv2.IMREAD_GRAYSCALE).astype(np.uint8)
imgR = cv2.imread(r'C:\Users\User\RaspberryPi4_PCfolder\stereo2-l.jpg',cv2.IMREAD_GRAYSCALE).astype(np.uint8)

stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
disparity = stereo.compute(imgL,imgR)
plt.imshow(disparity,'gray')
plt.show()

while True:
    # Display left and right images
    #cv2.imshow('left.png',imgL)
    #cv2.waitKey(0)
    #cv2.imshow('right.png',imgR)
    #cv2.waitKey(0)
    # Concatenate the two images horizontally
    combined_img = np.hstack((imgL, imgR))
    show_image(combined_img,"combined_img")
    find_matching_points(imgL,imgR)
    object_detection(imgL)


    # Display menu to choose preproceesing algorithm
    print("Choose deep mapping algorithm:")
    print("1. Image smoothing")
    print("2. Image thresholding")
    print("3. Blob detection")
    print("4. Edge detection")
    print("5. Image sharpening")
    print("6. Quit")
    choice = input("Enter your choice: ")

    # Apply preproceesing algorithm
    if choice == '6': # Quit
        break
    else:
        output = preproceesing(imgL, imgR, choice)


    # Show the output image
    if choice == '1':
        window_name = "Image Smoothing"
    elif choice == '2':
        window_name = "Image Thresholding"
    elif choice == '3':
        window_name = "Blob Detection"
    elif choice == '4':
        window_name = "Edge Detection"
    elif choice == '5':
        window_name = "Image Sharpening"

    show_image(output,window_name)

    # Display menu to choose Stereo algorithm
    print("Choose Stereo algorithm:")
    print("1. stereo_bm")
    print("2. stereo_bm_GaussianBlur")
    print("3. stereo_sgbm_wls")
    print("4. stereo_bm_noise_removal")
    print("5. Quit")
    choice = input("Enter your choice: ")

    # Apply Stereo algorithm
    if choice == '5': # Quit
        break
    else:
        output = stereo(imgL, imgR, choice)


    # Show the output image
    if choice == '1':
        window_name = "stereo_bm"
    elif choice == '2':
        window_name = "stereo_bm_GaussianBlur"
    elif choice == '3':
        window_name = "stereo_sgbm_wls"
    elif choice == '4':
        window_name = "stereo_bm_noise_removal"

    show_image(output,window_name)    

    # Wait for key press and handle exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Clean up resources
cv2.destroyAllWindows()


