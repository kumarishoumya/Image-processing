import numpy as np
import os
import cv2
from scipy.ndimage import variance
from skimage.color import rgb2gray
from skimage.filters import laplace
from skimage.transform import resize



sharp_laplaces= []
path = "C:/archive/blur_dataset_scaled/sharp"
for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img))
    img_array = resize(img_array, (400, 600))
    img_array = rgb2gray(img_array)
    edge_laplace = laplace(img_array, ksize=3)
    sharp_laplaces.append([variance(edge_laplace),np.amax(edge_laplace)])


blurry_laplaces = []
path = "C:/archive/blur_dataset_scaled/defocused_blurred"
for img in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, img))
    img_array = resize(img_array, (400, 600))
    img_array = rgb2gray(img_array)
    edge_laplace = laplace(img_array, ksize=3)
    blurry_laplaces.append([variance(edge_laplace), np.amax(edge_laplace)])


import pandas as pd
pd.DataFrame(np.array(sharp_laplaces)).to_csv("E:/aws/sharp.csv")
pd.DataFrame(np.array(blurry_laplaces)).to_csv("E:/aws/blur.csv")

