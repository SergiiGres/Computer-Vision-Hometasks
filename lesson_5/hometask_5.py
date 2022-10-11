import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import distance

plt.rcParams['figure.figsize'] = [15, 10]


def simple_quantization(image, color_palette):
    rows, cols, channels = image.shape
    result = np.zeros_like(img)

    for r in range(rows):
        for c in range(cols):
            # Extract the original pixel value
            pixel = list(img[r, c, :])

            # Find the closest colour from the pallette (using absolute value/Euclidean distance)
            # Note: You may need more than one line of code here
            dist = []
            for color in color_palette:
                dst = distance.euclidean(pixel, color)
                dist.append(dst)
            new_pixel = colors[np.argmin(dist)]

            # Apply quantization
            result[r, c, :] = new_pixel
    return result

def quantization_with_diffusion(image, color_palette):
    # Make a temporal copy of the original image, we will need it for error diffusion
    result = np.copy(image)
    dith = np.zeros_like(img)
    rows, cols, channels = result.shape
    for r in range(1, rows - 1):
        for c in range(1, cols - 1):
            # Extract the original pixel value
            pixel = list(result[r, c, :])
            # Find the closest colour from the pallette (using absolute value/Euclidean distance)
            # Note: You may need more than one line of code here
            dist = []
            for color in color_palette:
                dst = distance.euclidean(pixel, color)
                dist.append(dst)
            new_pixel = colors[np.argmin(dist)]

            # Apply quantization
            result[r, c, :] = new_pixel

            # Compute quantization error
            quant_error = pixel - new_pixel

            # Diffuse the quantization error accroding to the FS diffusion matrix
            # Note: You may need more than one line of code here
            result[r + 1][c] = result[r + 1][c] + quant_error * 7 / 16
            result[r - 1][c + 1] = result[r - 1][c + 1] + quant_error * 3 / 16
            result[r][c + 1] = result[r][c + 1] + quant_error * 5 / 16
            result[r + 1][c + 1] = result[r + 1][c + 1] + quant_error * 1 / 16

            # Apply dithering
            dith[r, c, :] = new_pixel

    return result, dith


# Load image
img = cv2.imread('data/kodim23.png')

# Convert it to RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot it
plt.imshow(img)
plt.show()

# Black, dark gray, light gray, white
colors = np.array([[0, 0, 0],
                   [64, 64, 64],
                   [192, 192, 192],
                   [255, 255, 255]])

# Cast the image to float
img = img.astype(np.float64)

# Prepare for quantization
# Apply quantization

# Show quantized image (don't forget to cast back to uint8)
quantized = simple_quantization(img, colors).astype(np.uint8)
plt.imshow(quantized), plt.title('Simple quantization')
plt.show()

# Compute average quantization error
avg_quant_error = np.mean(np.subtract(img, quantized))
print(avg_quant_error)

# Show quantized image (don't forget to cast back to uint8)
img_tmp, dithering = quantization_with_diffusion(img, colors)

img_tmp = img_tmp.astype(np.uint8)
dithering = dithering.astype(np.uint8)
plt.subplot(121), plt.imshow(img_tmp), plt.title('Black, dark gray, light gray, white')  # optimally quantized
plt.subplot(122), plt.imshow(dithering)  # dithering
plt.show()

# Black and white
colors = np.array([[0, 0, 0],
                   [255, 255, 255]])

# Show quantized image (don't forget to cast back to uint8)
img_tmp, dithering = quantization_with_diffusion(img, colors)

img_tmp = img_tmp.astype(np.uint8)
dithering = dithering.astype(np.uint8)
plt.subplot(121), plt.imshow(img_tmp), plt.title('Black and white')  # optimally quantized
plt.subplot(122), plt.imshow(dithering)  # dithering
plt.show()

# Compute average quantization error for dithered image
avg_dith_error = np.mean(np.subtract(quantized, dithering))

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4).fit(np.reshape(img, (-1, 1)))
colors = kmeans.cluster_centers_

img_tmp, dithering = quantization_with_diffusion(img, colors)

# Show quantized image (don't forget to cast back to uint8)
img_tmp = img_tmp.astype(np.uint8)
dithering = dithering.astype(np.uint8)
plt.subplot(121), plt.imshow(img_tmp), plt.title('Optimally quantized')  # optimally quantized
plt.subplot(122), plt.imshow(dithering)  # dithering
plt.show()
