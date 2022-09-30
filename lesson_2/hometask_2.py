import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [15, 5]


def white_patch():
    image = cv2.imread('data/sea.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image)
    # Define white patch and the coefficients
    row, col = 485, 864
    white = image[row, col, :]
    coefficients = 255.0 / white

    # Apply white balancing and generate balanced image
    result = np.zeros_like(image, dtype=np.float32)
    for channel in range(3):
        result[..., channel] = image[..., channel] * coefficients[channel]

    # White patching does not guarantee that the dynamic range is preserved, images must be clipped.
    result = result / 255
    result[result > 1] = 1
    plt.subplot(121), plt.imshow(image)
    plt.subplot(122), plt.imshow(result)
    plt.show()


def grey_world(image):
    # Compute the mean values for all three colour channels (red, green, blue)
    # red, green, blue = cv2.split(image)
    mean_r = np.mean(image[..., 0])
    mean_g = np.mean(image[..., 1])
    mean_b = np.mean(image[..., 2])

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
    result = np.zeros_like(img, dtype=np.float32)
    for channel in range(3):
        result[..., channel] = np.minimum(img[..., channel] * coeffs[channel], 255)
    result = result / 255
    result[result > 1] = 1

    return result


def scale_by_max(image):
    # Compute the maximum values for all three colour channels (red, green, blue)
    red, green, blue = cv2.split(image)
    max_r = np.max(red)
    max_g = np.max(green)
    max_b = np.max(blue)

    red = red / max_r
    green = green / max_g
    blue = blue / max_b

    # Apply color balancing and generate the balanced image
    result = cv2.merge([red, green, blue])

    return result


# White patch
white_patch()

# Gray world
# Load your image
img = cv2.imread('data/dark.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
balanced = grey_world(img)
# Show the original and the balanced image side by side
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(balanced)
plt.show()

# Scale-by-max
# Load your image
img = cv2.imread('data/dark.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
balanced = scale_by_max(img)

# Show the original and the balanced image side by side
plt.subplot(121), plt.imshow(img)
plt.subplot(122), plt.imshow(balanced)
plt.show()
