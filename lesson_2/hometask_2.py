import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [15, 5]

img = cv2.imread('data/sea.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

# Define white patch and the coefficients
row, col = 485, 864
white = img[row, col, :]
coeffs = 255.0 / white

# Apply white balancing and generate balanced image
balanced = np.zeros_like(img, dtype=np.float32)
for channel in range(3):
    balanced[..., channel] = img[..., channel] * coeffs[channel]

# White patching does not guarantee that the dynamic range is preserved, images must be clipped.
balanced = balanced / 255
balanced[balanced > 1] = 1

plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(balanced)
plt.show()


# Gray world
# Load your image
img = cv2.imread('data/dark.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Compute the mean values for all three colour channels (red, green, blue)
red, green, blue = cv2.split(img)
mean_r = np.average(red)
mean_g = np.average(green)
mean_b = np.average(blue)

# Compute the coefficients kr, kg, kb
# Note: there are 3 coefficients to compute but we only have 2 equations.
# Therefore, you have to make an assumption, fix the value of one of the
# coefficients and compute the remaining two
# Hint: You can fix the coefficient of the brightest colour channel to 1.
gray = np.mean([mean_r, mean_g, mean_b])
kr = gray / mean_r
kg = gray / mean_g
kb = gray / mean_b
coeffs = [kr, kg, kb]
print('coefficients (kr, kg, kb) = ({}, {}, {})'.format(kr, kg, kb))

# Apply color balancing and generate the balanced image
balanced = np.zeros_like(img, dtype=np.float32)
for channel in range(3):
    balanced[..., channel] = np.minimum(img[..., channel] * coeffs[channel], 255)

balanced = balanced / 255
balanced[balanced > 1] = 1
# balanced = grey_world(img)

# Show the original and the balanced image side by side
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(balanced)
plt.show()
