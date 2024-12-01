import torch
import numpy as np
import sys
import torch.nn as nn
import torch.nn.functional as F
from utils import egb
import mymodels.UNet as Unet
import utils.my_dataset as dataset
# import utils.make_graph as Graph
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from skimage import measure,graph
import cv2
from scipy.spatial.distance import cdist

class FinetuneLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward_dist(self,result,f_map,vit_T):
        
        B,C,W,H=f_map.shape
        f_map=f_map.permute(0,2,3,1).view(B,W*H,C)
        Cos=f_map@f_map.permute(0,2,1)
        Cos = torch.clamp(Cos, min=-1+1e-7, max=1-1e-7)
        Dot=1-2*torch.sigmoid(-torch.sqrt(2-2*Cos))
        mask=torch.zeros_like(result)
        neg_mask=torch.zeros_like(result)
        b,c,w,h=result.shape
        for i in range(b):
            mask[i]=torch.where(result[i]>vit_T[i],torch.ones_like(result[i]),torch.zeros_like(result[i]))
            neg_mask[i]=torch.where(result[i]<=vit_T[i],torch.ones_like(result[i]),torch.zeros_like(result[i]))
        mask=mask.view(b,w*h,c)
        neg_mask=neg_mask.view(b,w*h,c)
        mask=mask@mask.permute(0,2,1)
        neg_mask=neg_mask@neg_mask.permute(0,2,1)
        # pos_index=torch.where(mask==1)
        Dot_pos=Dot*mask
        Dot_neg=Dot*neg_mask
        # neg_index=torch.where(neg_mask==1)
        # print(Dot.shape)
        loss_pos=Dot_pos.sum(dim=(-1,-2))+Dot_neg.sum(dim=(-1,-2))
        # loss_pos=Dot[pos_index].sum(dim=-1).sum(dim=-1)+Dot[neg_index].sum(dim=(-1,-2))
        loss_all=Dot.sum(dim=(-1,-2))
        loss=loss_pos/loss_all
        # print(loss)
        return loss
    def forward(self,X,f_map,threshold,predict):
        b,l=threshold.shape
        #0.005,0.01,0.1,0.25,0.5
        loss=self.forward_dist(X,f_map,threshold[:,0])*predict[:,0]+self.forward_dist(X,f_map,threshold[:,1])*predict[:,1] \
                +self.forward_dist(X,f_map,threshold[:,2])*predict[:,2]+self.forward_dist(X,f_map,threshold[:,3])*predict[:,3]+\
                self.forward_dist(X,f_map,threshold[:,4])*predict[:,4]
        
        
        return loss.mean()
        