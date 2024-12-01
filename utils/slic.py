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
class SLIC(object):
    def __init__(self) -> None:
        super().__init__()
    def read_path(self,read_path,save_seg_path,annotations_file,save_seglab_path,save_imglab_path):
        img_names = pd.read_csv(annotations_file,header=None)
        #遍历该目录下的所有图片文件
        for i in range(len(img_names)):
            img_path = os.path.join(read_path, img_names.iloc[i, 0])
            print(img_path)
            img = Image.open(img_path)
            # w,h=img.size
            img=img.resize((128,128))
            img=np.array(img)
            w,h,c=img.shape
            segment = slic(img,n_segments=64,max_num_iter=10,convert2lab=True,enforce_connectivity=True,compactness=50)
            seg_map = segment.flatten()
            # img=img.reshape(-1,3)
            # seg_lab = [np.where(seg_map == u_label)[0]
            #             for u_label in np.unique(seg_map)]
            # img_lab = [img[np.where(seg_map == u_label)[0]].flatten()
            #             for u_label in np.unique(seg_map)]
            # seg_lab=np.array(seg_lab,dtype=object)
            # img_lab = np.array(img_lab,dtype=object)
            # max_len = np.max([len(a) for a in img_lab])

            # result=np.asarray([np.pad(a, (0, max_len - len(a)), 'mean') for a in img_lab])
            # print(np.min(segment))
            if np.min(segment)==0:
                print('zero!!!!')
                break
            properties=measure.regionprops(segment)
            img=img.reshape(w,h,c)
            img_with_boundaries = mark_boundaries(img, segment, color=[0,0,1])
            new_save_seg_path=save_seg_path+"/"+img_names.iloc[i, 0][:-4]+'.npy'
            print(new_save_seg_path)
            np.save(new_save_seg_path, segment)
            # np.save(save_seglab_path+"/"+img_names.iloc[i, 0][:-4]+'.npy',seg_map)
            # np.save(save_imglab_path+"/"+img_names.iloc[i, 0][:-4]+'.npy',result)
            cv2.imwrite("/home/XJF/code/HRSID/results/Seg_broundarie/"+img_names.iloc[i, 0],img_with_boundaries*255)

    def get_segment(self,path):
        return np.load(path,allow_pickle=True) 

    def my_slic(self,segment):
        labels=measure.regionprops(segment)
        num_sup=len(labels)
        sup_pix=np.zeros(len(labels))
        sup_centre=np.zeros((len(labels),2))
        for i in range(len(labels)):
            sup_pix[i]=labels[i].area            
        return sup_centre.astype(int),sup_pix.astype(int),num_sup

if __name__=='__main__':
    
    S=SLIC()
   
    # S.read_path("/home/XJF/code/SSDD_Seg/data/train/img",
    #             save_seg_path="/home/XJF/code/SSDD_Seg/data/SupMasks",
    #             save_seglab_path="/home/XJF/code/SSDD_Seg/data/SegLab",
    #             save_imglab_path="/home/XJF/code/SSDD_Seg/data/ImgLab",
    #             annotations_file="/home/XJF/code/SSDD_Seg/data/train.txt")
    # S.read_path("/home/XJF/code/SSDD_Seg/data/test/img",
    #             save_seg_path="/home/XJF/code/SSDD_Seg/data/SupMasks",
    #             save_seglab_path="/home/XJF/code/SSDD_Seg/data/SegLab",
    #             save_imglab_path="/home/XJF/code/SSDD_Seg/data/ImgLab",
    #             annotations_file="/home/XJF/code/SSDD_Seg/data/test.txt")
    # S.read_path("/home/XJF/code/HRSID/data/train/targets",
    #             save_seg_path="/home/XJF/code/HRSID/data/SupMasks",
    #             save_seglab_path="/home/XJF/code/HRSID/data/SegLab",
    #             save_imglab_path="/home/XJF/code/HRSID/data/ImgLab",
    #             # save_index_path="/home/XJF/SAR_Ground_Truth/data/BSR/BSDS500/SupIndexs",
    #             annotations_file="/home/XJF/code/HRSID/data/train.txt")
    S.read_path("/home/XJF/code/HRSID/data/test/img",
                save_seg_path="/home/XJF/code/HRSID/data/SupMasks",
                save_seglab_path="/home/XJF/code/HRSID/data/SegLab",
                save_imglab_path="/home/XJF/code/HRSID/data/ImgLab",
                # save_index_path="/home/XJF/SAR_Ground_Truth/data/BSR/BSDS500/SupIndexs",
                annotations_file="/home/XJF/code/HRSID/data/test.txt")
    # seg_lab=S.get_segment("/home/XJF/code/SSDD_Seg/data/SegLab/000004Target01.npy")
    # print(seg_lab.shape)
    # img_names = pd.read_csv("/home/XJF/code/SSDD_Seg/data/train.txt",header=None)
        #遍历该目录下的所有图片文件
    W=0
    # for i in range(len(img_names)):
    #     path="/home/XJF/code/SSDD_Seg/data/ImgLab"+"/"+img_names.iloc[i, 0][:-4]+'.npy'
    #     # img_path = os.path.join(read_path, img_names.iloc[i, 0])
    #     img_lab=S.get_segment(path)
    #     w,h=img_lab.shape
    #     W+=h
    # print(W/len(img_names))