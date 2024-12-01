import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image
from sklearn.cluster import KMeans
names = os.listdir("/home/XJF/code/SSDD_Seg/data/train/img")
def read_path(read_path,annotations_file,save_t_path,save_img_path,save_img_path2,save_img_path3):
    img_names = pd.read_csv(annotations_file,header=None)
        #遍历该目录下的所有图片文件
    for i in range(len(img_names)):
        img_path = os.path.join(read_path, img_names.iloc[i, 0])
        print(img_path)
        img = Image.open(img_path)
        
        # img=img.resize((128,128))
        image = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)  
        w,h,_=image.shape
        image=image[15:image.shape[0]-15,15:image.shape[1]-15]
        # img=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image=cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        # print(image.shape)
        # print(image[0].shape)
        threshold,result = cv2.threshold(image[:,:,0],0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        th2 = cv2.adaptiveThreshold(image[:,:,0],255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
        init=np.array([[0,0,0],[255,255,255]])
        k_means = KMeans(n_clusters=2, random_state=1,init=init)
        k_data = image.reshape(-1,3)
        k_means.fit(k_data)
        y_predict = k_means.predict(k_data).reshape(image.shape[0],image.shape[1])
        # print(y_predict.max)
        # break
        # img.save(img_path)
        print(result.shape,threshold)
        # np.save(os.path.join(save_t_path,img_names.iloc[i, 0][:-4])+".npy",threshold)
        cv2.imwrite(os.path.join(save_img_path,img_names.iloc[i, 0]),result)
        cv2.imwrite(os.path.join(save_img_path2,img_names.iloc[i, 0]),th2)
        cv2.imwrite(os.path.join(save_img_path3,img_names.iloc[i, 0]),y_predict*255)
        
# read_path("/home/XJF/code/SSDD_Seg/data/train/img","/home/XJF/code/SSDD_Seg/data/train.txt","/home/XJF/code/SSDD_Seg/data/train/threshold","/home/XJF/code/SSDD_Seg/results/fig_result_t")

read_path("/home/XJF/code/SSDD_Seg/data/test/img","/home/XJF/code/SSDD_Seg/data/test.txt","/home/XJF/code/SSDD_Seg/data/test/threshold","/home/XJF/code/SSDD_Seg/results/fig_result_t","/home/XJF/code/SSDD_Seg/results/fig_result_adapt","/home/XJF/code/SSDD_Seg/results/fig_result_clu")
