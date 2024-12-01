import os
import sys
sys.path.append('..')
import pandas as pd

from random import randint

from torchvision.io import read_image
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import cv2
# import utils.egb as EGB
import utils.slic
# import slic
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torchvision.transforms import ToTensor
from torchvision import transforms
class MYDataset(Dataset):
    def __init__(self, annotations_file, img_dir,threshold_dirs,transform=None, target_transform=None):
        self.img_names = pd.read_csv(annotations_file,header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.threshold_dirs=threshold_dirs
        # self.index_dir=index_dir

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names.iloc[idx, 0])
        thresholds=np.zeros(len(self.threshold_dirs))
        for i in range(len(self.threshold_dirs)):
            threshold_path= os.path.join(self.threshold_dirs[i], self.img_names.iloc[idx, 0][:-4])+'.npy'
            thresholds[i]=np.load(threshold_path)

        
        image=Image.open(img_path)
        
        image1 = np.asarray(image)
        image1=image1[15:image1.shape[0]-15,15:image1.shape[1]-15]
        image1=Image.fromarray(image1)
        if self.transform:
            image = self.transform(image)
            image1=self.transform(image1)
        return image,image1,torch.tensor(thresholds/255,dtype=torch.float32)
        # return image, label,img_lab,seg_lab



if __name__=='__main__':
    thr_dirs=['/home/XJF/code/SSDD_Seg/data/train/threshold_0','/home/XJF/code/SSDD_Seg/data/train/threshold_1',
              '/home/XJF/code/SSDD_Seg/data/train/threshold_2','/home/XJF/code/SSDD_Seg/data/train/threshold_3',
              '/home/XJF/code/SSDD_Seg/data/train/threshold_4']
    transform = transforms.Compose([
        transforms.Resize([96,96]),
        transforms.ToTensor(),
        #transforms.Resize((512,512))
        # transforms.Normalize([0.432284, 0.4364989, 0.36451283], [ 0.24840787, 0.23247136, 0.24171728]),
        #transforms.Normalize([ 0.1583,  0.1583, 0.1583], [ 0.0885,  0.0885,  0.0885])
    ])
    training_data = MYDataset(
        annotations_file="/home/XJF/code/SSDD_Seg/data/train.txt",
        img_dir="/home/XJF/code/SSDD_Seg/data/train/img",
        threshold_dirs=thr_dirs,
        # index_dir='/home/XJF/SAR_Ground_Truth/data/BSR/BSDS500/SupIndexs',
        transform=transform
    )
    train_dataloader = DataLoader(training_data, batch_size=2, shuffle=True)
    print(len(train_dataloader))
    train_features, t= next(iter(train_dataloader))
    # print(path.shape)
    # print(train_features.shape)
    # b,c,w,h=train_features.shape
    # img = train_features.squeeze()
    # print(img.dtype)
    print(train_features.shape)
    print(t)