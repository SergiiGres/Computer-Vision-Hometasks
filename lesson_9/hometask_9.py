import cv2
import numpy as np
from matplotlib import pyplot as plt
import dlib

plt.rcParams['figure.figsize'] = [15, 10]

# img = cv2.imread('data/image2.jpg')  # one false detection
img = cv2.imread('data/image3.jpg')  # no false detections
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(img)
plt.show()

# Let's load the detector
detector = dlib.get_frontal_face_detector()
# Detect faces, see http://dlib.net/face_detector.py.html
# 1 --> upsampling factor
rects = detector(gray, 1)

print('Number of detected faces:', len(rects))
print(rects)
print(rects[0].left)


def rect_to_bb(rect):
    # Dlib rect --> OpenCV rect
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y

    return (x, y, w, h)


# Draw rectangle around each face
result_dlib = np.copy(img)
faces_dlib_img = []
for rect in rects:
    # Draw rectangle around the face
    x, y, w, h = rect_to_bb(rect)
    print(x, y, w, h)
    cv2.rectangle(result_dlib, (x, y), (x + w, y + h), (0, 255, 0), 3)
    faces_dlib_img.append(img[y:y + h, x:x + w, :])

plt.imshow(result_dlib), plt.title('dlib')
plt.show()
