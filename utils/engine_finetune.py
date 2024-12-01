import sys

from tqdm import tqdm
import torch
import numpy as np
torch.set_printoptions(threshold=np.inf)
from torch import distributed as dist
import utils.slic as slic
# import utils.make_graph as graph
import utils.merge_sup as merge_sup
import numpy as np
# import code.SSDD_Seg.metric as metric
import cv2
from skimage import color
from torchvision.transforms import ToTensor
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from Bio.Cluster import kcluster
from typing import Iterable

def train_one_epoch(model_vit:torch.nn.Module,
                    model_unet:torch.nn.Module,
                    dataloader:Iterable,
                    loss:torch.nn.Module,
                    optim:torch.optim.Optimizer,
                    device:torch.device,epoch:int,
                    args):
    model_unet.train()
    
    model_vit.train()
    
    tbar=tqdm(dataloader)
    Avg_loss=0.0
    Len=len(dataloader)
    for i, ( X_vit,X_unet, y) in enumerate(tbar):
        X_vit, y = X_vit.clone().cuda(device, non_blocking=True), y.clone().cuda(device, non_blocking=True)
        X_unet=X_unet.clone().cuda(device, non_blocking=True)
        # print(X_vit.device)
        # print(model_vit)
        predict=model_vit(X_vit)
        f_map=model_unet(X_unet)
        l=loss(X_unet[:,0:1,:,:],f_map,y,predict)

        l.backward()
        Avg_loss+=l.item()
        optim.step()
        optim.zero_grad()
        if torch.isnan(l):
            print("find Loss is Nan")
            return -1
        tbar.set_description(f'Train_Epoch [{epoch}/{args.epochs}]'
                            f"mean_loss: {Avg_loss/(i+1):.6f} "
                        )
    return 0
    
def test_one_epoch(model_vit:torch.nn.Module,
                    model_unet:torch.nn.Module,
                    dataloader:Iterable,
                    loss:torch.nn.Module,
                    device:torch.device,epoch:int,
                    args):
    model_unet.eval()
    model_vit.eval()
    tbar=tqdm(dataloader)
    Avg_loss=0.0
    Len=len(dataloader)
    with torch.no_grad():
        for i, ( X_vit,X_unet, y) in enumerate(tbar):

            X_vit, y = X_vit.clone().cuda(device, non_blocking=True), y.clone().cuda(device, non_blocking=True)
            X_unet=X_unet.clone().cuda(device, non_blocking=True)
            predict=model_vit(X_vit)
            f_map=model_unet(X_unet)
            l=loss(X_unet[:,0:1,:,:],f_map,y,predict)
            Avg_loss+=l.item()
            # print(y)
            # print(predict)
            B,C,W,H=f_map.shape
            if args.MCTM:
                Thr=(predict*y).sum(dim=1)
            else:
                tmp_t=torch.zeros_like(y)
                for jjj in range(y.shape[1]):
                    tmp_t[:,jjj]=50*((jjj))/255
                    if jjj==0:
                        tmp_t[:,jjj]=10/255
                Thr=(tmp_t*y).sum(dim=1)
            result=torch.zeros(X_unet.shape[0],X_unet.shape[2],X_unet.shape[3]).to(args.device)
            # c1=((f_map.permute(0,2,3,1)[:,0,0,:]+f_map.permute(0,2,3,1)[:,127,127,:]+f_map.permute(0,2,3,1)[:,127,0,:]+f_map.permute(0,2,3,1)[:,0,127,:])/4).cpu().numpy()
            # c2=((f_map.permute(0,2,3,1)[:,63,63,:]+f_map.permute(0,2,3,1)[:,63,64,:]+f_map.permute(0,2,3,1)[:,64,63,:]+f_map.permute(0,2,3,1)[:,64,64,:])/4).cpu().numpy()
            f_map=f_map.view(B,C,-1).permute(0,2,1)
            f_map=f_map.cpu().numpy()
            # result_kmeans=np.zeros((f_map.shape[0],f_map.shape[1]))
            for j in range(len(Thr)):
                # k_means=KMeans(n_clusters=2, random_state=10,init=np.array([c1[j],c2[j]]))
                # result_kmeans[j]=k_means.fit_predict(f_map[j])
                result[j]=torch.where(X_unet[j,0,:,:]>Thr[j],torch.ones_like(result[j]),torch.zeros_like(result[j]))
            result=result.cpu().numpy()
            # result_kmeans=result_kmeans.reshape(B,W,H)
            # result=result*result_kmeans
            for j in range (len(Thr)):
                final_result=result[j,:,:]
                final_result=final_result.astype(np.uint8)
                # final_result=cv2.medianBlur(final_result,23)
                # result[j,:,:]=result[j,:,:]*255
                # result[j,:,:]=cv2.medianBlur(result[j,:,:].astype(np.uint8),23)
                cv2.imwrite(args.fig_save_path+'/'+str(i*len(Thr)+j)+'.png',final_result*255)

            tbar.set_description(f'Test_Epoch [{epoch}/{args.epochs}]'
                                f"mean_loss: {Avg_loss/(i+1):.6f} "
                            )
        return Avg_loss/Len