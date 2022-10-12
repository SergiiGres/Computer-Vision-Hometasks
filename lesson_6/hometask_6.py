import cv2
import math
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [15, 10]

# Let's read the image
img = cv2.imread('data/document.jpg')
# Convert it to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray) / 255
rows, cols = gray.shape

# Let's plot the images (colour and gray scale)
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(gray, cmap='gray')
plt.show()

# Harris Corner Detector
# Let us now compute Harris corners. Remember that the Harris detector computes the "cornerness" score
# for each image pixel.

# Compute Harris corners (use the available OpenCV functions)
# Suggested parameters:
#            block size of 2 pixels
#            gradient kernel size of 3 pixels
#            k parameter equal to 0.04
cornerness = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# We are not interested in edges, so put to zero all negative cornerness values
cornerness.clip(min=0, out=cornerness)

# Since cornerness has a huge dynamic range, let's take the logarithm for better visualization and manipulation
cornerness = np.log(cornerness + 1e-6)

# Let's now plot the image and the corresponding Harris corners (in log scale)
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(cornerness)
plt.show()

# Detection thresholds
th_top_left, th_top_right = -1e6, -1e6
th_bottom_left, th_bottom_right = -1e6, -1e6

# Corner coordinates
opt_top_left, opt_top_right = None, None
opt_bottom_left, opt_bottom_right = None, None

# Size of each quadrant (in pixels)
quad_size = 7

# Let's now scan the Harris detection results
for r in range(quad_size, rows - quad_size):
    for c in range(quad_size, cols - quad_size):
        # Edges with too small cornerness score are discarded, -7 seems like a good value
        if cornerness[r, c] < -7:
            continue

        # Extract block consisting of 4 quadrants
        block = 255 * gray[r - quad_size:r + quad_size + 1, c - quad_size:c + quad_size + 1]

        # Extract the four quandrants
        quad_top_left = block[0:quad_size, 0:quad_size]
        quad_top_right = block[0:quad_size, quad_size+1:quad_size*2 + 1]
        quad_bottom_left = block[quad_size+1:quad_size*2+1, 0:quad_size]
        quad_bottom_right = block[quad_size+1:quad_size*2+1, quad_size+1:quad_size*2+1]

        # Top-left corner
        # For the top-left document corner, the bottom-right quadrant is mostly paper and the rest is
        # darker background. Therefore, I suggest the descriptor to be the average difference between
        # the paper quandrant and the sum of the 3 remaining bakcground quandrants
        descriptor = np.mean(quad_bottom_right) -\
                     np.mean(quad_top_left) - np.mean(quad_top_right) - np.mean(quad_bottom_left)

        # # Let's detect the best descriptor
        if descriptor > th_top_left:
            # We update the threshold
            th_top_left = descriptor
            # And we update the optimal location
            opt_top_left = (c, r)

        # Top-right corner
        # (your implementation goes here)
        descriptor = np.mean(quad_bottom_left) - \
                     np.mean(quad_top_left) - np.mean(quad_bottom_right) - np.mean(quad_top_right)

        if descriptor > th_top_right:
            # We update the threshold
            th_top_right = descriptor
            # And we update the optimal location
            opt_top_right = (c, r)

        # Bottom-left corner
        # (your implementation goes here)
        descriptor = np.mean(quad_top_right) - \
                     np.mean(quad_top_left) - np.mean(quad_bottom_right) - np.mean(quad_bottom_left)

        if descriptor > th_bottom_left:
            # We update the threshold
            th_bottom_left = descriptor
            # And we update the optimal location
            opt_bottom_left = (c, r)

        # Bottom-right corner
        # (your implementation goes here)
        descriptor = np.mean(quad_top_left) - \
                     np.mean(quad_bottom_right) - np.mean(quad_bottom_left) - np.mean(quad_top_right)

        if descriptor > th_bottom_right:
            # We update the threshold
            th_bottom_right = descriptor
            # And we update the optimal location
            opt_bottom_right = (c, r)

# Let's draw circles at the detected corners
out = cv2.circle(img, opt_top_left, 3, (255, 0, 0), -1)
out = cv2.circle(img, opt_top_right, 3, (255, 0, 0), -1)
out = cv2.circle(img, opt_bottom_left, 3, (255, 0, 0), -1)
out = cv2.circle(img, opt_bottom_right, 3, (255, 0, 0), -1)

# out = cv2.drawMarker(img,opt_top_left,(255,0,0),cv2.MARKER_SQUARE ,20,1,cv2.LINE_AA)
# out = cv2.drawMarker(img,opt_top_left,(255,0,0),cv2.MARKER_CROSS ,20,1,cv2.LINE_AA)

# And finally we plot the images (with the detected document corners)
plt.subplot(121), plt.imshow(out)
plt.subplot(122), plt.imshow(cornerness)
plt.show()
