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

img = cv2.imread('../data/document.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img, cmap='gray')

# First, let's have a look at the histogram.
h = np.histogram(img, 256)
plt.bar(h[1][0:-1], h[0])
plt.xlabel('Colour'), plt.ylabel('Count')
plt.grid(True)

#Otsu Thresholding
# Let's now implement the Otsu thresholding algorithm. Remember that the algorithm consists of an optimization process
# that finds the thresholds that minimizes the intra-class variance or, equivalently, maximizes the inter-class variance.
#
# In this homework, you are going to demonstrate the working principle of the Otsu algorithm. Therefore, you won't have
# to worry about an efficient implementation, we are going to use the brute force approach here.

# Get image dimensions
rows, cols =
# Compute the total amount of image pixels
num_pixels =

# Initializations
best_wcv = 1e6  # Best within-class variance (wcv)
opt_th = None   # Threshold corresponding to the best wcv

# Brute force search using all possible thresholds (levels of gray)
for th in range(0, 256):
    # Extract the image pixels corresponding to the background
    foreground =
    # Extract the image pixels corresponding to the background
    background =

    # If foreground or background are empty, continue
    if len(foreground) == 0 or len(background) == 0:
        continue

    # Compute class-weights (omega parameters) for foreground and background
    omega_f =
    omega_b =

    # Compute pixel variance for foreground and background
    # Hint: Check out the var function from numpy ;-)
    # https://numpy.org/doc/stable/reference/generated/numpy.var.html
    sigma_f =
    sigma_b =

    # Compute the within-class variance
    wcv =

    # Perform the optimization
    if wcv < best_wcv:
        best =
        opt_th =

# Print out the optimal threshold found by Otsu algorithm
print('Optimal threshold', opt_th)

# Finally, let's compare the original image and its thresholded representation.
plt.subplot(121), plt.imshow(img, cmap='gray')
plt.subplot(122), plt.imshow(img > opt_th, cmap='gray')

# Questions
# Looking at the computed histogram, could it be considered bimodal?
# Looking at the computed histogram, what binarization threshold would you chose? Why?
# Looking at the resulting (thresholded) image, is the text binarization (detection) good?