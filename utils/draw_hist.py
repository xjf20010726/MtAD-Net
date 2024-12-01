import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.data import astronaut
from skimage import color,measure
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from scipy.spatial import distance #导包
import torch
import os
import pandas as pd
from PIL import Image
from skimage import color,graph
from sklearn.cluster import MeanShift,KMeans

def draw_hist(img_path,save_path):
    image=cv2.imread(img_path,0)
    clutter1=image[0:15,:]
    clutter2=image[image.shape[0]-15:image.shape[0],:]
    clutter3=image[15:image.shape[0]-15,0:15]
    clutter4=image[15:image.shape[0]-15,image.shape[1]-15:image.shape[1]]

    clutter1=clutter1.flatten()
    clutter2=clutter2.flatten()
    clutter3=clutter3.flatten()
    clutter4=clutter4.flatten()
    clutter=np.concatenate((clutter1,clutter2,clutter3,clutter4),axis=0)
    clutter=np.sort(clutter)
    hist, bin_edges=np.histogram(clutter,bins=256,range=(0,255),density=True)
    new=np.cumsum(hist)
    print(hist[100])
    plt.hist(clutter,bins=256,density=True,range=(0,255),cumulative=False,color='b')
    plt.plot([100,100],[-0,10*hist[100]],color='r',linestyle='--')

    new_clutter=image[15:image.shape[0]-15,15:image.shape[1]-15]
    new_clutter=new_clutter.flatten()
    # plt.hist(new_clutter,bins=256,density=True,range=(0,255),cumulative=False,color='g')
    # plt.plot([0,100],[new[100],new[100]],color='b',linestyle='--')
    ax = plt.gca()
    # 移到原点
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['left'].set_position(('data', 0))

    plt.savefig(save_path)
draw_hist("/home/XJF/code/SSDD_Seg/data/test/img/000001Target01.png",'/home/XJF/code/SSDD_Seg/output.png')