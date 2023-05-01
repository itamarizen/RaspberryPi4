# ---------------------------- imports ----------------------------- #
import sys,os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

# add base directory to sys.path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# ------------------------------------------------------------------ #

# ---------------------------- globals ----------------------------- #
filter = None

# ------------------------------------------------------------------ #

# ---------------------------- functios ---------------------------- #
def preprocessing(img, ksize=11, filter=None):
    if filter == 'median':
        img = cv2.medianBlur(img, ksize)
    elif filter == 'bilateral':
        img = cv2.bilateralFilter(img, 9, 75, 75)
    elif filter == 'gaussian':
        img = cv2.GaussianBlur(img, (5, 5), 0)
    elif filter == 'mean':
        img = cv2.blur(img, (5, 5))
    elif filter == 'sobel':
        img = cv2.Sobel(img, cv2.CV_8U, 1, 1, ksize=3)
    elif filter == 'laplacian':
        img = cv2.Laplacian(img, cv2.CV_8U, ksize=3)
    elif filter == 'canny':
        sigma = 0.3 # 30 % up&down
        median = np.median(img)
        lower = int(min(0,(1.0 - sigma) * median))
        upper = int(max(255,(1.0 + sigma) * median))
        img = cv2.Canny(img, lower, upper)
    elif filter == 'threshold':
        _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    elif filter == 'adaptive_threshold':
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    elif filter == 'otsu':
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    elif filter == 'erode':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.erode(img, kernel, iterations=1)
    elif filter == 'dilate':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.dilate(img, kernel, iterations=1)
    elif filter == 'opening':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif filter == 'closing':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif filter == 'tophat':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    elif filter == 'blackhat':
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    else:
        raise ValueError('Invalid filter')
    
    return img

def get_corespondence_matching(img1,img2,display = False):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if(display):
        # Visualize keypoints
        imgSift = cv2.drawKeypoints(
            img1, kp1, None)
        # cv2.imshow("SIFT Keypoints", imgSift)
        plt.imshow(imgSift, cmap = "gray"),plt.axis('off'),plt.title("Visualize keypoints of Scale-Invariant Feature Transform")

    # Match keypoints in both images
    # Based on: https://docs.opencv2.org/master/dc/dc3/tutorial_py_matcher.html
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Keep good matches: calculate distinctive image features
    # Lowe, D.G. Distinctive Image Features from Scale-Invariant Keypoints. International Journal of Computer Vision 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
    # https://www.cs.ubc.ca/~lowe/papers/ijcv204.pdf
    matchesMask = [[0, 0] for i in range(len(matches))]
    good = []
    pts1 = []
    pts2 = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            # Keep this keypoint pair
            matchesMask[i] = [1, 0]
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    # Draw the keypoint matches between both pictures
    # Still based on: https://docs.opencv2.org/master/dc/dc3/tutorial_py_matcher.html
    draw_params = dict(matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    matchesMask=matchesMask[300:1700],
                    flags=cv2.DrawMatchesFlags_DEFAULT)

    keypoint_matches = cv2.drawMatchesKnn(
        img1, kp1, img2, kp2, matches[300:1700], None, **draw_params)
    
    if(display):
        plt.imshow(keypoint_matches, cmap = "gray"),plt.axis('off'),plt.title("keypoint_matches")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    return pts1,pts2

def stereo_rectification(img1, img2,pts1,pts2,display = False):
    # STEREO RECTIFICATION
    # Calculate the fundamental matrix for the cameras
    # https://docs.opencv2.org/master/da/de9/tutorial_py_epipolar_geometry.html
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    fundamental_matrix, inliers = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # We select only inlier points
    pts1 = pts1[inliers.ravel() == 1]
    pts2 = pts2[inliers.ravel() == 1]

    # Visualize epilines
    # Adapted from: https://docs.opencv2.org/master/da/de9/tutorial_py_epipolar_geometry.html


    def drawlines(img1src, img2src, lines, pts1src, pts2src):
        ''' img1 - image on which we draw the epilines for the points in img2
            lines - corresponding epilines '''
        r, c = img1src.shape
        img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
        img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
        # Edit: use the same random seed so that two images are comparable!
        np.random.seed(0)
        for r, pt1, pt2 in zip(lines, pts1src, pts2src):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
            img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
            img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
            img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
        return img1color, img2color


    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(
        pts2.reshape(-1, 1, 2), 2, fundamental_matrix)
    lines1 = lines1.reshape(-1, 3)
    img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(
        pts1.reshape(-1, 1, 2), 1, fundamental_matrix)
    lines2 = lines2.reshape(-1, 3)
    img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)
    if(display):
        plt.subplot(121), plt.imshow(img5)
        plt.subplot(122), plt.imshow(img3)
        plt.suptitle("Epilines in both images")
        plt.show()


    # Stereo rectification (uncalibrated variant)
    # Adapted from: https://stackoverflow.com/a/62607343
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    _, H1, H2 = cv2.stereoRectifyUncalibrated(
        np.float32(pts1), np.float32(pts2), fundamental_matrix, imgSize=(w1, h1)
    )

    # Rectify (undistort) the images and save them
    # Adapted from: https://stackoverflow.com/a/62607343
    img1_rectified = cv2.warpPerspective(img1, H1, (w1, h1))
    img2_rectified = cv2.warpPerspective(img2, H2, (w2, h2))

    if(display):
        # Draw the rectified images
        fig, axes = plt.subplots(1, 2, figsize=(15, 10))
        axes[0].imshow(img1_rectified, cmap="gray")
        axes[1].imshow(img2_rectified, cmap="gray")
        axes[0].axhline(250)
        axes[1].axhline(250)
        axes[0].axhline(450)
        axes[1].axhline(450)
        plt.suptitle("Rectified images")
        plt.show()
        cv2.waitKey()
        cv2.destroyAllWindows()

    return img1_rectified,img2_rectified

def get_disparity(img1_rectified,img2_rectified,display = False):
    # CALCULATE DISPARITY (DEPTH MAP)
    # Adapted from: https://github.com/opencv2/opencv2/blob/master/samples/python/stereo_match.py
    # and: https://docs.opencv2.org/master/dd/d53/tutorial_py_depthmap.html

    # StereoSGBM Parameter explanations:
    # https://docs.opencv2.org/4.5.0/d2/d85/classcv2_1_1StereoSGBM.html

    # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
    block_size = 11
    min_disp = -128
    max_disp = 128
    # Maximum disparity minus minimum disparity. The value is always greater than zero.
    # In the current implementation, this parameter must be divisible by 16.
    num_disp = max_disp - min_disp
    # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
    # Normally, a value within the 5-15 range is good enough
    uniquenessRatio = 5
    # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
    # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    speckleWindowSize = 50
    # Maximum disparity variation within each connected component.
    # If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16.
    # Normally, 1 or 2 is good enough.
    speckleRange = 1
    disp12MaxDiff = 0

    stereo = cv2.StereoSGBM_create(
    minDisparity=min_disp,
    numDisparities=num_disp,
    blockSize=block_size,
    uniquenessRatio=uniquenessRatio,
    speckleWindowSize=speckleWindowSize,
    speckleRange=speckleRange,
    disp12MaxDiff=disp12MaxDiff,
    P1=8 * 1 * block_size * block_size,
    P2=32 * 1 * block_size * block_size,
    )
    disparity_SGBM = stereo.compute(img1_rectified, img2_rectified)

    # Normalize the values to a range from 0..255 for a grayscale image
    disparity_SGBM = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                beta=0, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    if(display):
        plt.imshow(disparity_SGBM, cmap='gray');plt.suptitle("disparity_SGBM")
        plt.show()
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    return disparity_SGBM

def extract_closest_objects(disparity, display=False):
    # Compute the histogram of the disparity values
    histogram, bins = np.histogram(disparity, bins=256,normed=True)
    
    # remove total black (backgraund)
    histogram[0] = 0
    '''
    distance: This parameter controls the minimum horizontal distance between peaks. A larger value will result in fewer peaks being detected, while a smaller value will detect more peaks. A good starting point could be to set distance to a fraction of the total number of bins in the histogram, such as distance = len(bins) // 20.

    prominence: This parameter controls the minimum height difference between a peak and its surrounding valleys. A larger value will result in fewer peaks being detected, while a smaller value will detect more peaks. A good starting point could be to set prominence to a fraction of the maximum histogram value, such as prominence = histogram.max() // 10.

    width: This parameter controls the minimum horizontal width of a peak. A larger value will result in wider peaks being detected, while a smaller value will detect narrower peaks. A good starting point could be to set width to a fraction of the total number of bins in the histogram, such as width = len(bins) // 50.

    height: This parameter controls the minimum height of a peak. A larger value will result in taller peaks being detected, while a smaller value will detect shorter peaks. A good starting point could be to set height to a fraction of the maximum histogram value, such as height = histogram.max() // 20.

    threshold: This parameter controls the minimum absolute height of a peak. A larger value will result in fewer peaks being detected, while a smaller value will detect more peaks. A good starting point could be to set threshold to a fraction of the maximum histogram value, such as threshold = histogram.max() // 50.
    '''
    plateau_size = 1
    height = np.max(histogram) * 0.1
    threshold = None
    distance = 5
    prominence = np.max(histogram) * 0.05
    width = None
    peaks, _ = find_peaks(histogram,plateau_size=plateau_size,height=height,threshold=threshold, distance=distance,prominence=prominence,width=width)
    
    if display:
        plt.subplot(1, 2, 1)
        plt.plot(histogram);plt.title("histogram")
        plt.subplot(1, 2, 2)
        plt.plot(peaks, histogram[peaks], "xr"); plt.plot(histogram); plt.legend(['peaks']);plt.title("analyzed histogram")
        plt.axhline(height, linestyle="--", color="gray")
        # Plot vertical lines at the midpoint between adjacent peaks
        colors = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'pink', 'brown', 'gray', 'teal']
        for i in range(len(peaks)):
            color = colors[i % len(colors)]
            plt.axvline(peaks[i] - distance, linestyle="--", color=color)
            plt.axvline(peaks[i] + distance, linestyle="--", color=color)
        plt.suptitle("Histogram analysis")
        plt.show()
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        kernel = np.ones((5,5),np.uint8)
        for i in range(len(peaks)):
            min_val, max_val = peaks[-1-i]-(distance), peaks[-1-i]+(distance)
            obj = cv2.normalize(cv2.threshold(disparity, min_val, max_val, cv2.THRESH_TOZERO)[1], None, 0, 255, cv2.NORM_MINMAX)
            obj[(disparity < min_val) | (disparity > max_val)] = 0
            closing = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, kernel)
            opening = cv2.morphologyEx(obj, cv2.MORPH_OPEN, kernel)
            dilation = cv2.dilate(obj,kernel,iterations = 1)
            # Create a figure with three columns and one row
            fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

            # Display the original image in the first subplot
            axs[0,0].imshow(obj, cmap='gray')
            axs[0,0].set_title(f'Object {i+1} - Non operated')

            # Display the closing image in the second subplot
            axs[0,1].imshow(closing, cmap='gray')
            axs[0,1].set_title(f'Object {i+1} - Closing')

            # Display the opening image in the third subplot
            axs[1,0].imshow(opening, cmap='gray')
            axs[1,0].set_title(f'Object {i+1} - Opening')

            # Display the dilation image in the fourth subplot
            axs[1,1].imshow(dilation, cmap='gray')
            axs[1,1].set_title(f'Object {i+1} - Dilation')

            # Show the figure
            plt.show()        
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    # ---------------------- script start here ------------------------- #

    imgL = cv2.imread('./images/L.png',cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread('./images/R.png',cv2.IMREAD_GRAYSCALE)

    # imgL_filtered = preprocessing(imgL,filter='median',display=True)
    # imgR_filtered = preprocessing(imgR, filter='gaussian',display=True)

    pts1,pts2 = get_corespondence_matching(imgL,imgR,display=True)

    imgL_rectified,imgR_rectified = stereo_rectification(imgL,imgR,pts1,pts2,display=False)
    
    disparity = get_disparity(imgL_rectified,imgR_rectified,display=False)

    eco = extract_closest_objects(disparity,display = True)


if __name__ == '__main__':
    main()