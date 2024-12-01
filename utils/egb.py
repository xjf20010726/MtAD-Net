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
class EGB(object):
    def __init__(self) -> None:
        super().__init__()
    def read_path(self,read_path,save_path,annotations_file):
        img_names = pd.read_csv(annotations_file,header=None)
        #遍历该目录下的所有图片文件
        for i in range(len(img_names)):
            img_path = os.path.join(read_path, img_names.iloc[i, 0])
            print(img_path)
            if '.jpg' in img_path:
                img = Image.open(img_path)
            else:
                img = Image.open(img_path+'.jpg')
            img=img.resize((512,512))
            img=np.array(img)
            segment = felzenszwalb(img, scale=1024, sigma=0.95, min_size=5)
            MaxNum=np.max(segment)
            segment=np.where(segment>0,segment,MaxNum+1)
            #####save figure

            new_save_path=save_path+"/"+img_names.iloc[i, 0]
            if '.jpg' in img_path:
                new_save_path=save_path+"/"+img_names.iloc[i, 0][:-4]+'.npy'
            else:
                new_save_path=save_path+"/"+img_names.iloc[i, 0]+'.npy'
            print(new_save_path)
            np.save(new_save_path, segment)

    def get_segment(self,path):
        return np.load(path)

    def egb(self,segment):
        # segment = felzenszwalb(img, scale=150, sigma=0.95, min_size=5)
        # MaxNum=np.max(segment)
        # segment=np.where(segment>0,segment,MaxNum+1)
        # print(np.sum(segment==0))
        labels=measure.regionprops(segment)
        num_sup=len(labels)
        sup_pix=np.zeros(len(labels))
        sup_centre=np.zeros((len(labels),2))
        for i in range(len(labels)):
            sup_pix[i]=labels[i].area
            index_s=np.argwhere(segment==labels[i].label)
            l=len(index_s)//2
            #print(labels[i].centroid)
            #print(sup_centre[i])
            sup_centre[i]=index_s[l]
            #print(labels[i].centroid,)
            
        return sup_centre.astype(int),sup_pix.astype(int),num_sup
    
class SLIC(object):
    def __init__(self) -> None:
        super().__init__()
    def read_path(self,read_path,save_seg_path,annotations_file):
        img_names = pd.read_csv(annotations_file,header=None)
        #遍历该目录下的所有图片文件
        for i in range(len(img_names)):
            img_path = os.path.join(read_path, img_names.iloc[i, 0])
            print(img_path)
            img = Image.open(img_path)
            w,h=img.size
            if w<64 or h<64:
                if w>h:
                    w=int(w*64/h)
                    h=64
                else:
                    h=int(h*64/w)
                    w=64
            img=img.resize((w,h))
            img=np.array(img)
            segment = slic(img,n_segments=128,max_num_iter=10,convert2lab=True,enforce_connectivity=True,compactness=50)
            # print(np.min(segment))
            if np.min(segment)==0:
                print('zero!!!!')
                break
            properties=measure.regionprops(segment)
            img_with_boundaries = mark_boundaries(img, segment, color=[0,0,1])
            new_save_seg_path=save_seg_path+"/"+img_names.iloc[i, 0][:-4]+'.npy'
            print(new_save_seg_path)
            np.save(new_save_seg_path, segment)
            cv2.imwrite("/home/XJF/SSDD_Seg/results/Seg_broundarie/"+img_names.iloc[i, 0],img_with_boundaries*255)

    def get_segment(self,path):
        return np.load(path)

    def my_slic(self,segment):
        labels=measure.regionprops(segment)
        num_sup=len(labels)
        sup_pix=np.zeros(len(labels))
        sup_centre=np.zeros((len(labels),2))
        for i in range(len(labels)):
            sup_pix[i]=labels[i].area            
        return sup_centre.astype(int),sup_pix.astype(int),num_sup
class Threshold(object):
    def __init__(self) -> None:
        super().__init__()
    def read_path(self,read_path,save_seg_path,annotations_file):
        img_names = pd.read_csv(annotations_file,header=None)
        #遍历该目录下的所有图片文件
        for i in range(len(img_names)):
            img_path = os.path.join(read_path, img_names.iloc[i, 0])
            print(img_path)
            img = cv2.imread(img_path,0)
           
            img=cv2.resize(img,(128,128))
           
            _,gt_threshold = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
            print(gt_threshold.shape)
            # print(np.min(segment))
            
            
            
            new_save_seg_path=save_seg_path+"/"+img_names.iloc[i, 0]
            
            print(new_save_seg_path)
            cv2.imwrite(new_save_seg_path, gt_threshold)
            # np.save(new_save_seg_path, gt_threshold)
            # cv2.imwrite("/home/XJF/Seg_broundarie/"+img_names.iloc[i, 0],img_with_boundaries*255)

class my_slic(object):
    def __init__(self) -> None:
        super().__init__()
    def read_path(self,read_path,save_seg_path,annotations_file):
        img_names = pd.read_csv(annotations_file,header=None)
        #遍历该目录下的所有图片文件
        for i in range(len(img_names)):
            img_path = os.path.join(read_path, img_names.iloc[i, 0])
            print(img_path)
            img = cv2.imread(img_path,1)
            img=cv2.resize(img,(128,128))
            segment = slic(img,n_segments=4,max_num_iter=100,convert2lab=True,enforce_connectivity=True,compactness=1)
            data_gt=color.label2rgb(segment)
            new_save_seg_path=save_seg_path+"/"+img_names.iloc[i, 0]
            
            print(new_save_seg_path)
            cv2.imwrite(new_save_seg_path, data_gt*255)
            # cv2.imwrite("/home/XJF/Seg_broundarie/"+img_names.iloc[i, 0],img_with_boundaries*255)
            
class my_egb(object):
    def __init__(self) -> None:
        super().__init__()
    def read_path(self,read_path,save_seg_path,annotations_file):
        img_names = pd.read_csv(annotations_file,header=None)
        #遍历该目录下的所有图片文件
        for i in range(len(img_names)):
            img_path = os.path.join(read_path, img_names.iloc[i, 0])
            print(img_path)
            img = cv2.imread(img_path,1)
            img=cv2.resize(img,(128,128))
            segment = felzenszwalb(img, scale=256, sigma=0.95, min_size=2)
            data_gt=color.label2rgb(segment)
            new_save_seg_path=save_seg_path+"/"+img_names.iloc[i, 0]
            
            print(new_save_seg_path)
            cv2.imwrite(new_save_seg_path, data_gt*255)

class kmeans(object):
    def __init__(self) -> None:
        super().__init__()
    def read_path(self,read_path,save_seg_path,annotations_file):
        img_names = pd.read_csv(annotations_file,header=None)
        #遍历该目录下的所有图片文件
        for i in range(len(img_names)):
            img_path = os.path.join(read_path, img_names.iloc[i, 0])
            print(img_path)
            img = cv2.imread(img_path,1)
            img=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img=cv2.resize(img,(128,128))
            print(img.shape)
            #break
            data = []

            for x in range(128):
                for y in range(128):
                    l,a,b = img[x,y,0],img[x,y,1],img[x,y,2]
                    data.append([l,a,b])

            Data = np.asarray(data)

            model=KMeans(n_clusters=2,max_iter=5,n_init='auto',init='random')
            segment = model.fit_predict(Data)
            segment=segment.reshape(128,128)
            print(segment.shape)
            #break
            data_gt=color.label2rgb(segment)
            new_save_seg_path=save_seg_path+"/"+img_names.iloc[i, 0]
            
            print(new_save_seg_path)
            cv2.imwrite(new_save_seg_path, data_gt*255)
class mean_shift(object):
    def __init__(self) -> None:
        super().__init__()
    def read_path(self,read_path,save_seg_path,annotations_file):
        img_names = pd.read_csv(annotations_file,header=None)
        #遍历该目录下的所有图片文件
        for i in range(len(img_names)):
            img_path = os.path.join(read_path, img_names.iloc[i, 0])
            print(img_path)
            img = cv2.imread(img_path,0)
            # img=cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            img=cv2.resize(img,(128,128))
            print(img.shape)
            #break
            # data = []

            # for x in range(128):
            #     for y in range(128):
            #         a,b = img[x,y,1],img[x,y,2]
            #         data.append([a,b])

            # Data = np.asarray(data)

            model=MeanShift(bandwidth=10,max_iter=10)
            segment = model.fit_predict(img.reshape(-1,1))
            segment=segment.reshape(128,128)
            print(segment.shape)
            #break
            data_gt=color.label2rgb(segment)
            new_save_seg_path=save_seg_path+"/"+img_names.iloc[i, 0]
            
            print(new_save_seg_path)
            cv2.imwrite(new_save_seg_path, data_gt*255)
if __name__=='__main__':
    
    # E=EGB()
    S=SLIC()
    T=Threshold()
    s=my_slic()
    e=my_egb()
    k=kmeans()
    m=mean_shift()
    # segments,label,sup_centre,sup_pix=E.egb(scale=256, sigma=0.8, min_size=10)
    # print(len(label))
    # #print(l)
    # labels=color.label2rgb(segments)
    # print(sup_centre)
    #print(len(np.where(segments==0)))
    #cv2.imshow("segment",labels)
    #cv2.waitKey(0) #等待按键
    S.read_path("/home/XJF/SSDD_Seg/data/train/img",
                save_seg_path="/home/XJF/SSDD_Seg/data/SupMasks",
                # save_index_path="/home/XJF/SAR_Ground_Truth/data/BSR/BSDS500/SupIndexs",
                annotations_file="/home/XJF/SSDD_Seg/data/train.txt")
    S.read_path("/home/XJF/SSDD_Seg/data/test/img",
                save_seg_path="/home/XJF/SSDD_Seg/data/SupMasks",
                # save_index_path="/home/XJF/SAR_Ground_Truth/data/BSR/BSDS500/SupIndexs",
                annotations_file="/home/XJF/SSDD_Seg/data/test.txt")
    
    # T.read_path("/home/XJF/SAR_Ground_Truth/data/PolSAR/test_data/Targets",
    #             save_seg_path="/home/XJF/SAR_Ground_Truth/results/Threshold",
    #             # save_index_path="/home/XJF/SAR_Ground_Truth/data/BSR/BSDS500/SupIndexs",
    #             annotations_file="/home/XJF/SAR_Ground_Truth/data/PolSAR/test_data.txt")
    # img=cv2.imread("/home/XJF/SAR_Ground_Truth/data/PolSAR/test_data/Targets/s2Target29.png",1)
    # img=cv2.imread("/home/XJF/SAR_Ground_Truth/data/PolSAR/train_data/Targets/09_01Target11.png",1)
    # img=cv2.resize(img,(128,128))
    # fig=np.load("/home/XJF/SAR_Ground_Truth/data/PolSAR/train_data/SupMasks/09_01Target11.npy")
    # fig=np.load("/home/XJF/SAR_Ground_Truth/data/PolSAR/test_data/SupMasks/s2Target29.npy")
    # col=color.label2rgb(fig)
    # img_with_boundaries = mark_boundaries(img, fig, color=[0,0,1])
    # rag=graph.rag_mean_color(img, fig)
    # graph.show_rag(fig, rag, img)
    # print(lc)
    # properties = measure.regionprops(fig)
    # print(len(properties))
    # cv2.imwrite("test.jpg",img_with_boundaries*255)
    # cv2.imwrite("test1.jpg",lc)
    # cv2.waitKey(0)
    # s.read_path("/home/XJF/SAR_Ground_Truth/data/PolSAR/test_data/Targets",
    #             save_seg_path="/home/XJF/SAR_Ground_Truth/results/slic",
    #             # save_index_path="/home/XJF/SAR_Ground_Truth/data/BSR/BSDS500/SupIndexs",
    #             annotations_file="/home/XJF/SAR_Ground_Truth/data/PolSAR/test_data.txt")
    # e.read_path("/home/XJF/SAR_Ground_Truth/data/PolSAR/test_data/Targets",
    #             save_seg_path="/home/XJF/SAR_Ground_Truth/results/egb",
    #             # save_index_path="/home/XJF/SAR_Ground_Truth/data/BSR/BSDS500/SupIndexs",
    #             annotations_file="/home/XJF/SAR_Ground_Truth/data/PolSAR/test_data.txt")
    # k.read_path("/home/XJF/SAR_Ground_Truth/data/PolSAR/test_data/Targets",
    #             save_seg_path="/home/XJF/SAR_Ground_Truth/results/kmeans",
    #             # save_index_path="/home/XJF/SAR_Ground_Truth/data/BSR/BSDS500/SupIndexs",
    #             annotations_file="/home/XJF/SAR_Ground_Truth/data/PolSAR/test_data.txt")
    # m.read_path("/home/XJF/SAR_Ground_Truth/data/PolSAR/test_data/Targets",
    #             save_seg_path="/home/XJF/SAR_Ground_Truth/results/Mshift",
    #             # save_index_path="/home/XJF/SAR_Ground_Truth/data/BSR/BSDS500/SupIndexs",
    #             annotations_file="/home/XJF/SAR_Ground_Truth/data/PolSAR/test_data.txt")