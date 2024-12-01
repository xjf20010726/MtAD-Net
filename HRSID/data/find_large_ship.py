import torch
import torch.nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import color,measure
import os
from skimage import measure,color
from skimage.segmentation import felzenszwalb
import scipy.io as io
import pandas as pd
np.set_printoptions(threshold=np.inf)
img_names = pd.read_csv("/home/XJF/code/SSDD_Seg/data/test.txt",header=None)
for i in range(len(img_names)):
    gt_path = os.path.join("/home/XJF/code/SSDD_Seg/data/test/mask", img_names.iloc[i, 0])
        
    gt_data=cv2.imread(gt_path,0)
    w,h=gt_data.shape
    if (w-30)*(h-30)>8000:
        print(img_names.iloc[i, 0])