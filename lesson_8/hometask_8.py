# In this homework you are going to implement your first machine learning algorithm to automatically binarize
# document images. The goal of document binarization is to seprate the characters (letters) from everything else.
# This is the crucial part for automatic document understanding and information extraction from the . In order to do
# so, you will use the Otsu thresholding algorithm.
#
# At the end of this notebook, there are a couple of questions for you to answer.

import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [15, 10]

# Let's load the document image we will be working on in this homework.

img = cv2.imread('data/document.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img, cmap='gray')
plt.show()

# First, let's have a look at the histogram.
h = np.histogram(img, 256)
plt.bar(h[1][0:-1], h[0])
plt.xlabel('Colour'), plt.ylabel('Count')
plt.grid(True)
plt.show()

# Otsu Thresholding
#
# Let's now implement the Otsu thresholding algorithm. Remember that the algorithm consists of an
# optimization process that finds the thresholds that minimizes the intra-class variance or, equivalently,
# maximizes the inter-class variance.
#
# In this homework, you are going to demonstrate the working principle of the Otsu algorithm. Therefore, you won't have
# to worry about an efficient implementation, we are going to use the brute force approach here.

# Get image dimensions
rows, cols = img.shape
# Compute the total amount of image pixels
num_pixels = rows * cols

# Initializations
best_wcv = 1e6  # Best within-class variance (wcv)
opt_th = None  # Threshold corresponding to the best wcv

# Brute force search using all possible thresholds (levels of gray)
for th in range(0, 256):
    # Extract the image pixels corresponding to the background
    foreground = img[img < th]
    # Extract the image pixels corresponding to the background
    background = img[img > th]

    # If foreground or background are empty, continue
    if len(foreground) == 0 or len(background) == 0:
        continue

    # Compute class-weights (omega parameters) for foreground and background
    omega_f = len(foreground) / len(img.flatten())
    omega_b = len(background) / len(img.flatten())

    # Compute pixel variance for foreground and background
    # Hint: Check out the var function from numpy ;-)
    # https://numpy.org/doc/stable/reference/generated/numpy.var.html
    sigma_f = np.var(foreground)
    sigma_b = np.var(background)

    # Compute the within-class variance
    wcv = omega_f * sigma_f * sigma_f + omega_b * sigma_b * sigma_b

    # Perform the optimization
    if wcv < best_wcv:
        best_wcv = wcv
        opt_th = th

# Print out the optimal threshold found by Otsu algorithm
print('Optimal threshold', opt_th)


# Solution found on the internet
def threshold_otsu_impl(image, nbins=0.1):
    # validate grayscale
    if len(image.shape) == 1 or len(image.shape) > 2:
        print("must be a grayscale image.")
        return

    # validate multicolored
    if np.min(image) == np.max(image):
        print("the image must have multiple colors")
        return

    all_colors = image.flatten()
    total_weight = len(all_colors)
    least_variance = -1
    least_variance_threshold = -1

    # create an array of all possible threshold values which we want to loop through
    color_thresholds = np.arange(np.min(image) + nbins, np.max(image) - nbins, nbins)

    # loop through the thresholds to find the one with the least within class variance
    for color_threshold in color_thresholds:
        bg_pixels = all_colors[all_colors < color_threshold]
        weight_bg = len(bg_pixels) / total_weight
        variance_bg = np.var(bg_pixels)

        fg_pixels = all_colors[all_colors >= color_threshold]
        weight_fg = len(fg_pixels) / total_weight
        variance_fg = np.var(fg_pixels)

        within_class_variance = weight_fg * variance_fg + weight_bg * variance_bg
        if least_variance == -1 or least_variance > within_class_variance:
            least_variance = within_class_variance
            least_variance_threshold = color_threshold

    print("Optimal threshold by Otsu function:", least_variance_threshold)

    return least_variance_threshold


opt_th_func = threshold_otsu_impl(img)

# Finally, let's compare the original image and its thresholded representation.
plt.subplot(131), plt.title('Original'), plt.imshow(img, cmap='gray')
plt.subplot(132), plt.title('Otsu brute force'), plt.imshow(img > opt_th, cmap='gray')
plt.subplot(133), plt.title('Otsu function'), plt.imshow(img > opt_th_func, cmap='gray')
plt.show()

# Questions
# Q. Looking at the computed histogram, could it be considered bimodal?
# A. Seems yes.
# Q. Looking at the computed histogram, what binarization threshold would you chose? Why?
# A. Probably I didn't get the question, since I didn't chose binarization threshold. Could you please give me a
# clue what did I missed.
# Q. Looking at the resulting (thresholded) image, is the text binarization (detection) good?
# A. Yes, the text looks readable
