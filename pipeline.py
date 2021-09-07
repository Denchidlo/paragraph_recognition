import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from PIL import Image

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from umap import UMAP 

from scipy.spatial import distance_matrix
import cv2

img = cv2.imread('pics/image8.png', 0)

def black(img, threshold):
    img[img >= threshold] = 255
    img[img < threshold] = 0
    return img 

def inverse(img):
    return 255 - img

threshold = 200

mask = img.copy()

black(img, threshold)


smoothing_kernel = np.ones((7, 7), np.uint8)
smoothed = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, smoothing_kernel)
# smoothing_kernel = -np.ones((3, 3), np.uint8)
# smoothed = cv2.morphologyEx(smoothed, cv2.MORPH_GRADIENT, smoothing_kernel)

threshold = 50

mask = smoothed.copy()

black(mask, threshold)

final = inverse(mask)

cv2.imwrite('out/img_map.jpg', final)