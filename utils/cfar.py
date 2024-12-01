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

def cfar(read_path,annotations_file,save_t_path,save_img_path,Pro=0.01):
    img_names = pd.read_csv(annotations_file,header=None)
        #遍历该目录下的所有图片文件
    for i in range(len(img_names)):
        img_path = os.path.join(read_path, img_names.iloc[i, 0])
        # print(img_path)
        img = Image.open(img_path)
        image = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)  
        
        clutter1=image[0:15,:,0]
        clutter2=image[image.shape[0]-15:image.shape[0],:,0]
        clutter3=image[15:image.shape[0]-15,0:15,0]
        clutter4=image[15:image.shape[0]-15,image.shape[1]-15:image.shape[1],0]

        clutter1=clutter1.flatten()
        clutter2=clutter2.flatten()
        clutter3=clutter3.flatten()
        clutter4=clutter4.flatten()
        clutter=np.concatenate((clutter1,clutter2,clutter3,clutter4),axis=0)
        clutter=np.sort(clutter)
        # clutter=clutter[:int(1-np.sum(clutter>250)/len(clutter))*len(clutter)]
        if np.sum(clutter>250)>=0.5*len(clutter):
            clutter=clutter[:int(len(clutter)*0.5)]
        elif np.sum(clutter>250)>=0.4*len(clutter):
            clutter=clutter[:int(len(clutter)*0.6)]
        elif np.sum(clutter>250)>=0.3*len(clutter):
            clutter=clutter[:int(len(clutter)*0.7)]
        elif np.sum(clutter>250)>=0.2*len(clutter):
            clutter=clutter[:int(len(clutter)*0.8)]
        elif np.sum(clutter>250)>=0.1*len(clutter):
            clutter=clutter[:int(len(clutter)*0.9)]
        else:
            clutter=clutter[:int(len(clutter)*0.95)]
        # clutter=clutter[:int(len(clutter)*0.99)]
        hist, bin_edges=np.histogram(clutter,bins=256,range=(0,255),density=False)
        # print(hist,bin_edges.shape)
        # cdf = hist.cumsum()#累加频数得累计直方图
        # print(cdf)
        P=0.0
        T=0
        for j in range(255,-1,-1):
            # print(j)
            if P>Pro:
                T=j
                # if img_names.iloc[i, 0]=='000019Target01.png':
                print(T)
                break
            P+=hist[j]/len(clutter)
            
        
        # print(clutter.shape)
        threshold,result = cv2.threshold(image[:,:,0],T,255,cv2.THRESH_BINARY)
        if img_names.iloc[i, 0]=='000019Target01.png':
            print(T)
        np.save(os.path.join(save_t_path,img_names.iloc[i, 0][:-4])+".npy",threshold)
        cv2.imwrite(os.path.join(save_img_path,img_names.iloc[i, 0]),result)
        # if img_names.iloc[i, 0]=='000399Target03.png':
        #     print(image.shape)
        #     print(img_names.iloc[i, 0],result.shape)
        # break


        # img=img.resize((128,128))
        # image = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)  
        # # img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # # image=cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # # print(image.shape)
        # # print(image[0].shape)
        # threshold,result = cv2.threshold(image[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # th2 = cv2.adaptiveThreshold(image[:,:,0],255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        # init=np.array([[0,0,0],[255,255,255]])
        # k_means = KMeans(n_clusters=2, random_state=10,init=init)
        # k_data = image.reshape(-1,3)
        # k_means.fit(k_data)
        # y_predict = k_means.predict(k_data).reshape(128,128)
        # # print(y_predict.max)
        # # break
        # img.save(img_path)
        # print(result.shape,threshold)
        # np.save(os.path.join(save_t_path,img_names.iloc[i, 0][:-4])+".npy",threshold)
        # cv2.imwrite(os.path.join(save_img_path,img_names.iloc[i, 0]),result)
        # cv2.imwrite(os.path.join(save_img_path2,img_names.iloc[i, 0]),th2)
        # cv2.imwrite(os.path.join(save_img_path3,img_names.iloc[i, 0]),y_predict*255)


P=[0.005,0.01,0.1,0.25,0.5]
for p in range(len(P)):
    cfar("/home/XJF/code/HRSID/data/test/img","/home/XJF/code/HRSID/data/test.txt","/home/XJF/code/HRSID/data/test/threshold_"+str(p),"/home/XJF/code/HRSID/results/fig_results_cfar_"+str(p),P[p])
    cfar("/home/XJF/code/HRSID/data/train/img","/home/XJF/code/HRSID/data/train.txt","/home/XJF/code/HRSID/data/train/threshold_"+str(p),"/home/XJF/code/HRSID/results/fig_result_cfar_"+str(p),P[p])
# cfar("/home/XJF/code/HRSID/data/test/img","/home/XJF/code/HRSID/data/test.txt","/home/XJF/code/HRSID/data/test/threshold_3","/home/XJF/code/HRSID/results/fig_results_cfar_3")
# cfar("/home/XJF/code/HRSID/data/train/img","/home/XJF/code/HRSID/data/train.txt","/home/XJF/code/HRSID/data/train/threshold_3","/home/XJF/code/HRSID/results/fig_result_cfar_3")