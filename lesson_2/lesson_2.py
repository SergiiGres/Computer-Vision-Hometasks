import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [15, 5]
from time import time

img = cv2.imread('data/kodim05.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.show()

start = time()
rows, cols = img.shape
hist = np.zeros(256)
for r in range(rows):
    for c in range(cols):
        hist[img[r, c]] = hist[img[r, c]] + 1
print('Elapsed time:', time() - start)
plt.plot(np.arange(0, 256), hist)
plt.grid(True)
plt.xlabel('Pixel color'), plt.ylabel('Number of pixels')

# ------
cdf = np.zeros(256)
for idx, h in enumerate(hist):
    cdf[idx] = np.sum(hist[0:idx + 1])
cdf = cdf / np.sum(hist)
plt.plot(cdf), plt.grid(False)
plt.xlabel('Pixel color'), plt.ylabel('CDF')

equalized = np.zeros((rows, cols), dtype=np.uint8)
for r in range(rows):
    for c in range(cols):
        equalized[r, c] = 255 * cdf[img[r, c]]

plt.subplot(121), plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(122), plt.imshow(equalized, cmap='gray', vmin=0, vmax=255)
plt.show()

# -----
start = time()
hist, bins = np.histogram(img.ravel(), bins=256, range=(0, 255))
print('Elapsed time:', time() - start)
plt.plot(bins[0:-1] + 0.5, hist), plt.grid(False)

dst = cv2.equalizeHist(img)

plt.subplot(121), plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.subplot(122), plt.imshow(dst, cmap='gray', vmin=0, vmax=255)
plt.show()
