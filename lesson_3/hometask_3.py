import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [15, 10]

img = cv2.imread('data/kodim01.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

# Create a blurred (unsharp) version of the original image (you can use Gaussian blurring)
unsharp = cv2.GaussianBlur(img, ksize=(9, 9), sigmaX=10)
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(unsharp)
plt.show()

# Create the difference image (original − unsharp)
# Note: Remember that you are working with uint8 data types. Any addition or substractions
# might result in overflow or underflow, respectively. You can prevent this by casting the images to float.
diff = cv2.subtract(img, unsharp)
plt.imshow(diff)
plt.show()

# Apply USM to get the resulting image using `sharpened = original + (original − unsharp) × amount`
# Note: Again, take care of underflows/overflows if necessary.
diff_small = diff.copy()
for channel in range(3):
    diff_small[..., channel] = np.minimum(diff[..., channel] * 0.1, 255)
sharpened = cv2.add(img, diff_small)
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(sharpened)
plt.show()

diff_normal = diff.copy()
for channel in range(3):
    diff_normal[..., channel] = np.minimum(diff[..., channel] * 1.5, 255)
sharpened = cv2.add(img, diff_normal)
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(sharpened)
plt.show()

diff_high = diff.copy()
for channel in range(3):
    diff_high[..., channel] = np.minimum(diff[..., channel] * 10, 255)
sharpened = cv2.add(img, diff_high)
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(sharpened)
plt.show()

# Amount increases gamma. The reasonable value is 0.5 to 1.5, but it depends on what kind of result we whant to get.
# In case it is too small we get almost the same picture.
# In case it is too big we get whitening of photo
