# ---------------------------- imports ----------------------------- #
import sys,os
import numpy as np
import cv2
from matplotlib import pyplot as plt

# add base directory to sys.path
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# ------------------------------------------------------------------ #

# ---------------------------- globals ----------------------------- #
A = 1
# ------------------------------------------------------------------ #


# ---------------------------- functios ---------------------------- #

# PREPROCESSING
def preprocessing(img, ksize = 111):
    # img = cv2.medianBlur(img,ksize) 
    # img =  remove_uniform_areas(img)
    img = cv2.bilateralFilter(img, 5, 75, 75)
    return img   

def remove_uniform_areas(image):
    
    # Calculate the standard deviation of the grayscale image
    std_dev = np.std(image)
    
    # Create a binary mask with the same size as the image
    mask = np.zeros_like(image)
    
    # Find the pixels with a standard deviation greater than the mean
    mask[image > std_dev] = 255
    
    # Apply the mask to the input image
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    return masked_image

def histogram_equalization(img):
    
    # Calculate histogram
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    
    # Calculate cumulative distribution function
    cdf = hist.cumsum()
    
    # Normalize cdf
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    
    # Perform histogram equalization
    equalized = np.interp(img.flatten(), bins[:-1], cdf_normalized)
    equalized = equalized.reshape(img.shape).astype(np.uint8)
    
    # Calculate histogram of equalized image
    equalized_hist, equalized_bins = np.histogram(equalized.flatten(), 256, [0,256])
    
    # Calculate threshold values using Otsu's method
    _, low_thresh = cv2.threshold(equalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high_thresh = np.max(equalized_hist)
    low_thresh = np.min(equalized_hist)
    
    # Plot histograms and threshold values
    fig, axs = plt.subplots(1, 2)
    axs[0].hist(img.flatten(), 256, [0,256], color='b')
    axs[0].axvline(x=low_thresh, color='r')
    axs[0].axvline(x=high_thresh, color='r')
    axs[0].set_title('Original Image Histogram')
    axs[1].hist(equalized.flatten(), 256, [0,256], color='b')
    axs[1].axvline(x=low_thresh, color='r')
    axs[1].axvline(x=high_thresh, color='r')
    axs[1].set_title('Equalized Image Histogram')
    
    # Show images
    cv2.imshow('Original', img)
    cv2.imshow('Equalized', equalized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_image(image, windoe_name):
    # Create a named window with the WINDOW_AUTOSIZE flag
    cv2.namedWindow(windoe_name, cv2.WINDOW_NORMAL) #cv2.WINDOW_AUTOSIZE
    # Resize the window to match the image size
    cv2.resizeWindow(windoe_name, image.shape[1], image.shape[0])
    # Display the combined image
    cv2.imshow(windoe_name, image)
    cv2.waitKey(0)

def segment_and_label(disparity_image):

    # Threshold the disparity image to create a binary image
    _, binary_image = cv2.threshold(disparity_image, 0, 255, cv2.THRESH_BINARY)
    binary_image = binary_image.astype(np.uint8)
    
    # Perform morphological operations to remove noise and fill in gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    
    # Perform connected component analysis on the binary image
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8, ltype=cv2.CV_32S)
    
    # Create an output image for labeling the objects
    labeled_image = np.zeros_like(disparity_image)
    
    # Loop through the labels and draw the outlines of the objects in the output image
    for i in range(1, num_labels):
        object_mask = np.where(labels == i, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(labeled_image, contours, -1, (i*50, i*50, i*50), 2)
    
    return labeled_image, num_labels - 1


def outlining(image):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    erosion = cv2.erode(image,kernel,iterations = 1)
    outlined = cv2.bitwise_xor(erosion,image)
    return outlined 

def labeling(img):
    # global thresholding
    ret1,th1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    # Otsu's thresholding
    ret2,th2 = cv2.threshold(img.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret3,th3 = cv2.threshold(blur.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # plot all the images and their histograms
    images = [img, 0, th1,
            img, 0, th2,
            blur, 0, th3]
    titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
            'Original Noisy Image','Histogram',"Otsu's Thresholding",
            'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
    for i in range(3):
        plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
    plt.show()
# ------------------------------------------------------------------ #


# ---------------------- script start here ------------------------- #
if (A==1):
    # read two input images as grayscale images
    imgL = cv2.imread('./images/L.png',cv2.IMREAD_GRAYSCALE)
    imgL_prepro = preprocessing(imgL)
    #imgL_equalized = histogram_equalization(imgL)
    imgR = cv2.imread('./images/R.png',cv2.IMREAD_GRAYSCALE)
    imgR_prepro = preprocessing(imgR)
    #imgR_equalized = histogram_equalization(imgR)

if (A==2):
    # read two input images
    imgL = cv2.imread('aloeL.jpg',0)
    imgR = cv2.imread('aloeR.jpg',0)


stereo = cv2.StereoBM_create(numDisparities=16, blockSize=25)
stereo_prepro = cv2.StereoBM_create(numDisparities=16, blockSize=25)
#stereo_equalized = cv2.StereoBM_create(numDisparities=16, blockSize=25)

# compute the disparity map
disparity = stereo.compute(imgL,imgR)
disparity_prepro = stereo_prepro.compute(imgL_prepro,imgR_prepro)
labeled_image = outlining(disparity)
labeled_image = labeling(disparity)
show_image(labeled_image,'Labeled Image')
#disparity_equalized = stereo_prepro.compute(imgL_equalized,imgR_equalized)
plt.subplot(231), plt.imshow(imgL, cmap = "gray"),plt.axis('off')
plt.subplot(232), plt.imshow(imgR, cmap = "gray"),plt.axis('off')
plt.subplot(233), plt.imshow(disparity, cmap = "gray"),plt.axis('off'),plt.title("normal")
plt.subplot(234), plt.imshow(disparity_prepro, cmap = "gray"),plt.axis('off'),plt.title("prepro")
plt.subplot(235), plt.imshow(segment_and_label(labeled_image), cmap = "gray"),plt.axis('off'),plt.title("equalized")
#plt.subplot(235), plt.imshow(disparity_equalized, cmap = "gray"),plt.axis('off'),plt.title("equalized")
plt.imshow(disparity,'gray')
plt.show()
disparity.shape


# for aloe 
# # Initiate and StereoBM object
# stereo = cv2.StereoBM_create(numDisparities=128, blockSize=15)

# # compute the disparity map
# disparity = stereo.compute(imgL,imgR)
# #disparity1 = stereo.compute(imgR,imgL)
# plt.imshow(disparity,'gray')
# plt.show()



