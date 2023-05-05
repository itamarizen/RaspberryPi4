# ---------------------------- imports ----------------------------- #
import sys,os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.signal import find_peaks,gaussian
from skimage.measure import regionprops
from scipy.ndimage import label, find_objects

# add base directory to sys.path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# ------------------------------------------------------------------ #

# ---------------------------- globals ----------------------------- #
filter = None

# ------------------------------------------------------------------ #

# ---------------------------- functios ---------------------------- #
def preprocessing(image, filters=None, ksize=3, alpha=1.0, beta=0.0, gamma=0.0, display=False):
    filtered_img = image.copy()

    if filters is not None:
        for filter in filters:
            if filter == 'gaussian':
                filtered_img = cv2.GaussianBlur(filtered_img, (ksize, ksize), cv2.BORDER_DEFAULT)
            elif filter == 'median':
                filtered_img = cv2.medianBlur(filtered_img, ksize, cv2.BORDER_DEFAULT)
            elif filter == 'bilateral':
                filtered_img = cv2.bilateralFilter(filtered_img, ksize, 75, 75, cv2.BORDER_DEFAULT)
            elif filter == 'box':
                filtered_img = cv2.boxFilter(filtered_img, -1, (ksize, ksize), normalize=True, borderType=cv2.BORDER_DEFAULT)
            elif filter == 'sobelx':
                filtered_img = cv2.Sobel(filtered_img, cv2.CV_64F, 1, 0, ksize=ksize)
            elif filter == 'sobely':
                filtered_img = cv2.Sobel(filtered_img, cv2.CV_64F, 0, 1, ksize=ksize)
            elif filter == 'laplacian':
                filtered_img = cv2.Laplacian(filtered_img, cv2.CV_64F, ksize=ksize)
            elif filter == 'edge_enhance':
                kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
                filtered_img = cv2.filter2D(filtered_img, -1, kernel)
            elif filter == 'sharpen':
                kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                filtered_img = cv2.filter2D(filtered_img, -1, kernel)
            elif filter == 'contrast':
                filtered_img = cv2.convertScaleAbs(filtered_img, alpha=alpha, beta=beta)
            elif filter == 'brightness':
                filtered_img = cv2.convertScaleAbs(filtered_img, alpha=1.0, beta=gamma)
            else:
                raise ValueError('Invalid filter type')

    if display:
        plt.figure()
        plt.imshow(filtered_img, cmap='gray');plt.axis('off')
        plt.title(f"{filters} filter, ksize={ksize}")
        plt.show(block=False)

    return filtered_img

def get_correspondence_matching(img1,img2,display = False):
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    if(display):
        # Visualize keypoints
        imgSift = cv2.drawKeypoints(img1, kp1, None)
        # Show the figure
        plt.figure()
        plt.imshow(imgSift,cmap='tab20b');plt.title("Visualize keypoints of Scale-Invariant Feature Transform");plt.axis('off')
        plt.show(block=False)        

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
        plt.figure()
        plt.imshow(keypoint_matches, cmap = "gray"),plt.axis('off'),plt.title("keypoint_matches")
        plt.show(block=False)

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
        plt.figure()
        plt.subplot(121), plt.imshow(img5)
        plt.subplot(122), plt.imshow(img3)
        plt.suptitle("Epilines in both images")
        plt.show(block=False)

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
        plt.show(block=False)

    return img1_rectified,img2_rectified

def get_disparity(img1_rectified,img2_rectified,display = False):
    # CALCULATE DISPARITY (DEPTH MAP)
    # Adapted from: https://github.com/opencv2/opencv2/blob/master/samples/python/stereo_match.py
    # and: https://docs.opencv2.org/master/dd/d53/tutorial_py_depthmap.html

    # StereoSGBM Parameter explanations:
    # https://docs.opencv2.org/4.5.0/d2/d85/classcv2_1_1StereoSGBM.html

    # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
    block_size = 5
    min_disp = -128
    max_disp = 128
    # Maximum disparity minus minimum disparity. The value is always greater than zero.
    # In the current implementation, this parameter must be divisible by 16.
    num_disp = max_disp - min_disp
    # Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct.
    # Normally, a value within the 5-15 range is good enough
    uniquenessRatio = 10
    # Maximum size of smooth disparity regions to consider their noise speckles and invalidate.
    # Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
    speckleWindowSize = 100
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
    if display:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5), gridspec_kw={'width_ratios': [1, 2]})
        im = ax2.imshow(disparity_SGBM, cmap='gray')
        fig.colorbar(im, ax=ax2)
        ax2.set_title("disparity_SGBM")
        ax2.axis('off')
        ax1.plot([], [], label=f"block_size={block_size}")
        ax1.plot([], [], label=f"num_disp={num_disp}")
        ax1.plot([], [], label=f"uniquenessRatio={uniquenessRatio}")
        ax1.plot([], [], label=f"speckleWindowSize={speckleWindowSize}")
        ax1.legend(loc='center left')
        ax1.axis('off')
        plt.show(block=False)
    
    return disparity_SGBM

def enhance_disparity(disparity, display=False):
    # Define the kernel size for the weighted median filter
    kernel_size = 5

    # Compute the weights for each pixel in the local neighborhood
    weights = np.exp(-np.square(disparity - np.median(disparity))/kernel_size)

    # Apply the weighted median filter to the disparity map
    filtered_disparity = cv2.medianBlur(disparity, kernel_size, weights)

    # Step 1: Outlier detection
    min_disparity = -128
    max_disparity = 128
    weights = np.exp(-np.square(filtered_disparity - np.median(filtered_disparity))/kernel_size)
    filtered_disparity[filtered_disparity < min_disparity] = min_disparity
    filtered_disparity[filtered_disparity > max_disparity] = max_disparity

    # Step 2: Occlusion handling
    occlusion_mask = np.zeros_like(filtered_disparity)
    occlusion_mask[filtered_disparity == min_disparity] = 1
    occlusion_mask[filtered_disparity == max_disparity] = 1
    median_disparity = np.median(filtered_disparity)
    filtered_disparity[occlusion_mask == 1] = median_disparity

    # Step 3: Disparity edges refinement
    filtered_disparity = cv2.medianBlur(filtered_disparity, 5)
    filtered_disparity = cv2.bilateralFilter(filtered_disparity, 9, 75, 75)

    # Step 4: Uniform areas handling
    filtered_disparity = cv2.filter2D(filtered_disparity, -1, np.ones((5,5))/25)

    if display:
        # Display the final disparity map
        plt.figure()
        plt.imshow(filtered_disparity,cmap="gray"),plt.axis('off'),plt.title("Enhanced Disparity Map")
        plt.show(block=False)  

    return filtered_disparity

def extract_closest_objects_3(disparity, display=False):
    # Compute the histogram of the disparity values
    histogram, bins = np.histogram(disparity, bins=256, density=True)

    # Remove total black (background)
    histogram[0] = 0
    
    # Use peakutils to extract the peaks of the histogram
    from peakutils.peak import indexes
    prominence = np.max(histogram) * 0.05
    distance = 5
    peak_indices = indexes(histogram, thres=prominence, min_dist=distance)

    if display:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.plot(histogram)
        plt.title("Histogram")
        plt.subplot(1, 2, 2)
        plt.plot(peak_indices, histogram[peak_indices], "xr")
        plt.plot(histogram)
        plt.legend(['Peaks'])
        plt.title("Analyzed Histogram")
        plt.suptitle("Histogram Analysis")
        plt.show(block=False)

    # Use regionprops to extract the objects corresponding to each peak in the histogram
    objects = []
    for i, peak_index in enumerate(peak_indices[::-1]):
        # Define the minimum and maximum disparity values for the current object
        min_val, max_val = peak_index - distance, peak_index + distance

        # Extract the current object using thresholding and connected component analysis
        obj = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        obj[(disparity < min_val) | (disparity > max_val)] = 0
        thresh = cv2.threshold(obj, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        labels = label(thresh.astype(np.uint8))
        num_labels = np.max(labels)

        # Extract the regions corresponding to the current object
        for j in range(1, num_labels+1):
            mask = (labels == j).astype(np.uint8)
            props = regionprops(mask, intensity_image=obj)
            if len(props) == 0:
                continue
            # Choose the region with the highest mean intensity as the object
            max_mean_intensity_region = max(props, key=lambda x: x.mean_intensity)
            mask = max_mean_intensity_region.filled_image
            objects.append(mask)

        # Display the segmented object
        if display:
            plt.figure()
            plt.imshow(obj, cmap='gray')
            plt.title(f'Object {i+1}')
            plt.show(block=False)

    return objects

def extract_closest_objects(disparity, display=False):
    # Compute the histogram of the disparity values
    histogram, bins = np.histogram(disparity, bins=256,density=True)
    
    # remove total black (backgraund)
    histogram[0] = 0
    '''
    distance: This parameter controls the minimum horizontal distance between peaks. A larger value will result in fewer peaks being detected, while a smaller value will detect more peaks. A good starting point could be to set distance to a fraction of the total number of bins in the histogram, such as distance = len(bins) // 20.

    prominence: This parameter controls the minimum height difference between a peak and its surrounding valleys. A larger value will result in fewer peaks being detected, while a smaller value will detect more peaks. A good starting point could be to set prominence to a fraction of the maximum histogram value, such as prominence = histogram.max() // 10.

    width: This parameter controls the minimum horizontal width of a peak. A larger value will result in wider peaks being detected, while a smaller value will detect narrower peaks. A good starting point could be to set width to a fraction of the total number of bins in the histogram, such as width = len(bins) // 50.

    height: This parameter controls the minimum height of a peak. A larger value will result in taller peaks being detected, while a smaller value will detect shorter peaks. A good starting point could be to set height to a fraction of the maximum histogram value, such as height = histogram.max() // 20.

    threshold: This parameter controls the minimum absolute height of a peak. A larger value will result in fewer peaks being detected, while a smaller value will detect more peaks. A good starting point could be to set threshold to a fraction of the maximum histogram value, such as threshold = histogram.max() // 50.
    '''
    if not isinstance(disparity, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
        
    # Compute the histogram of the disparity values
    hist, bins = np.histogram(disparity, bins=256, density=True)
    
    # Remove total black (background)
    hist[0] = 0
    
    # Determine peak locations in the histogram
    peak_kwargs = {
        'plateau_size': 1,
        'height': np.max(hist) * 0.1,
        'threshold': None,
        'distance': len(bins) // 20,
        'prominence': np.max(hist) * 0.05,
        'width': len(bins) // 50
    }
    peaks, _ = find_peaks(hist, **peak_kwargs)
    
    if len(peaks) == 0:
        raise ValueError("No peaks found in histogram.")
    
    # Use regionprops to extract the objects corresponding to each peak in the histogram
    objects = []
    for i, peak in enumerate(reversed(peaks)):
        # Define the minimum and maximum disparity values for the current object
        min_val, max_val = peak - peak_kwargs['distance'], peak + peak_kwargs['distance']
        
        # Extract the current object using thresholding and connected component analysis
        obj = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        obj[(disparity < min_val) | (disparity > max_val)] = 0
        ret, thresh = cv2.threshold(obj, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        labels, num_labels = label(thresh.astype(np.uint8))
        
        if num_labels == 0:
            raise ValueError(f"No objects detected for peak {peak}.")
        
        # Extract the regions corresponding to the current object
        for j in range(1, num_labels + 1):
            mask = (labels == j).astype(np.uint8)
            props = regionprops(mask, intensity_image=obj)
            if len(props) == 0:
                continue
            # Choose the region with the highest mean intensity as the object
            max_mean_intensity_region = max(props, key=lambda x: x.mean_intensity)
            mask = max_mean_intensity_region.filled_image
            objects.append(mask)
        # Display the segmented object
        if display:
            plt.imshow(obj, cmap='gray')
            plt.title(f'Object {i+1}')
            plt.show(block=False)

    # if display:
    #     plt.subplot(1, 2, 1)
    #     plt.plot(histogram);plt.title("histogram")
    #     plt.subplot(1, 2, 2)
    #     plt.plot(peaks, histogram[peaks], "xr"); plt.plot(histogram); plt.legend(['peaks']);plt.title("analyzed histogram")
    #     plt.axhline(height, linestyle="--", color="gray")
    #     # Plot vertical lines at the midpoint between adjacent peaks
    #     colors = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'pink', 'brown', 'gray', 'teal']
    #     for i in range(len(peaks)):
    #         color = colors[i % len(colors)]
    #         plt.axvline(peaks[i] - distance, linestyle="--", color=color)
    #         plt.axvline(peaks[i] + distance, linestyle="--", color=color)
    #     plt.suptitle("Histogram analysis")
    #     plt.show(block=False)


    # # Use regionprops to extract the objects corresponding to each peak in the histogram
    # objects = []
    # for i in range(len(peaks)):
    #     # Define the minimum and maximum disparity values for the current object
    #     min_val, max_val = peaks[-1-i] - distance, peaks[-1-i] + distance

    #     # Extract the current object using thresholding and connected component analysis
    #     obj = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    #     obj[(disparity < min_val) | (disparity > max_val)] = 0
    #     ret, thresh = cv2.threshold(obj, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    #     labels = label(thresh.astype(np.uint8))
    #     num_labels = np.max(labels)

    #     # Extract the regions corresponding to the current object
    #     for j in range(1, num_labels+1):
    #         mask = (labels == j).astype(np.uint8)
    #         props = regionprops(mask, intensity_image=obj)
    #         if len(props) == 0:
    #             continue
    #         # Choose the region with the highest mean intensity as the object
    #         max_mean_intensity_region = max(props, key=lambda x: x.mean_intensity)
    #         mask = max_mean_intensity_region.filled_image
    #         objects.append(mask)
                
    #     # Display the segmented object
    #     if display:
    #         plt.figure()
    #         plt.imshow(obj, cmap='gray')
    #         plt.title(f'Object {i+1}')
    #         plt.show(block=False)

    # return objects
def read_images(pathL,pathR,display = False):
    imgL = cv2.imread(pathL,cv2.IMREAD_GRAYSCALE)
    imgR = cv2.imread(pathR,cv2.IMREAD_GRAYSCALE)
    if (display):
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(imgL,cmap="gray");plt.title("imgL");plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(imgR,cmap="gray");plt.title("imgR");plt.axis('off')
        plt.suptitle("input images")
        plt.show(block=False)        
    return imgL,imgR


def main():
    # ---------------------- script start here ------------------------- #

    imgL,imgR = read_images(pathL = './images/ruppinL.jpg', pathR = './images/ruppinR.jpg',display=False)

    imgL_filtered = preprocessing(imgL,filters=['median'],ksize=5,display=False)
    imgR_filtered = preprocessing(imgR, filters=['gaussian'],ksize=5,display=False)


    pts1,pts2 = get_correspondence_matching(imgL,imgR,display=False)
    pts1,pts2 = get_correspondence_matching(imgL_filtered,imgR_filtered,display=False)

    imgL_rectified,imgR_rectified = stereo_rectification(imgL,imgR,pts1,pts2,display=False)
    imgL_rectified_filtered,imgR_rectified_filtered = stereo_rectification(imgL_filtered,imgR_filtered,pts1,pts2,display=False)
    

    disparity = get_disparity(imgL_rectified,imgR_rectified,display=False)
    disparity_filtered = get_disparity(imgL_rectified_filtered,imgR_rectified_filtered,display=False)

    disparity_enhanced = enhance_disparity(disparity,display = False)

    eco = extract_closest_objects_3(disparity,display = True)
    eco_filtered = extract_closest_objects(disparity_filtered,display = True)


if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()


    '''
        imgR_filtered = preprocessing(imgR, filters=['gaussian','median','bilateral','box','sobelx',
                                                 'sobely','laplacian','edge_enhance','sharpen','contrast','brightness']
                                                 ,ksize=5, alpha=1.5, beta=0, gamma=50, display=True)


        # # display differences 
        # plt.figure()
        # plt.subplot(121), plt.imshow(img5)
        # plt.subplot(122), plt.imshow(img3)
        # plt.imshow(imgL_rectified-imgL_rectified_filtered,cmap='gray')
        # plt.suptitle("filtered VS non-filtered images results")
        # plt.show(block=False)


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
        plt.show(block=False)
    '''

