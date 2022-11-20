import os
import cv2
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [15, 10]

# Load the training labels
root = '/Users/sergiigres/PycharmProjects/ComputerVision/Computer-Vision-Hometasks/lesson_12/data/'  # Path to the dataset location, e.g., '/data/janko/dataset/GTSRB'
data = pd.read_csv(os.path.join(root, 'Train.csv'))

# Number of training samples (amount of samples in data)
num_samples = 39209

# Show random data samples
for ii in range(15):
    # Get random index
    idx = np.random.randint(0, num_samples)
    # Load image
    img = cv2.imread(os.path.join(root, data.iloc[idx]['Path']))
    # Convert image to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Show image
    plt.subplot(3, 5, ii + 1), plt.imshow(img), plt.title(data.iloc[idx]['ClassId'])
plt.show()

# Extract class identifiers
# Hint: Check the csv
ids = data['ClassId'].values

from collections import Counter
hist = Counter(ids)

plt.bar(hist.keys(), hist.values()), plt.grid(True)
plt.xlabel('Traffic Sign ID'), plt.ylabel('Counts')
plt.show()
