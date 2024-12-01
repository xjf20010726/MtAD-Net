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
class AverageMeter(object):
    """计算并存储均值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0 
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self.avg
    

def train(model,dataloader,loss,optim,device,epoch,writer,args):
    model.train()
    tbar=tqdm(dataloader)
    acc1=AverageMeter()
    E=slic.SLIC()
    c=torch.ones([1,1])
    c[0,0]=args.c
    c=c.cuda(device=args.device)
    for i, ( X, y,img_lab,seg_lab) in enumerate(tbar):
        segment=torch.squeeze(y).numpy()
        #print(X.dtype)
        X, y,img_lab = X.clone().cuda(device, non_blocking=True), y.clone().cuda(device, non_blocking=True),img_lab.clone().cuda(device, non_blocking=True)
        #获取超像素中心坐标点、超像素分割结构每个区域的像素个数、超像素总数目
        sup_centre,sup_pix,sup_num=E.my_slic(segment=segment)
        
        #获取模型输出
        pred_y,Gap_y,feature_map=model(X)
        #计算损失函数
        l=loss(sup_num,X,segment,feature_map,sup_pix,pred_y,Gap_y,c,seg_lab,args).sum()
        #计算预测准确率，后续换成计算评价指标
        loss_train=acc1.update(l)
        # with torch.autograd.detect_anomaly():
        l.backward()
        optim.step()
        
 
 
        # print("###### end optime step")
        optim.zero_grad()
        if torch.isnan(l):
            print(torch.isnan(torch.sum(torch.log(pred_y))))
            print(torch.isnan(torch.sum(feature_map)))
            print(torch.isnan(torch.sum(Gap_y)))
            print("Nan")
            return -1
        tbar.set_description(f'Train_Epoch [{epoch}/{args.epochs}]')
        tbar.set_postfix(loss='{0:1.3f}'.format(loss_train.item()))
    writer.add_scalar('Loss/Train_Avg_Loss', loss_train.item(), epoch)
    return 0
        
def train_v(model,dataloader,loss,optim,device,epoch,writer,args):
    model.train()
    tbar=tqdm(dataloader)
    acc1=AverageMeter()
    E=slic.SLIC()
    c=torch.ones([1,1])
    c[0,0]=args.c
    c=c.to(device=args.device)
    for i, ( X, y,img_lab,seg_lab) in enumerate(tbar):
        segment=torch.squeeze(y).numpy()
        #print(X.dtype)
        X, y,img_lab = X.clone().to(device, non_blocking=True), y.clone().to(device, non_blocking=True),img_lab.clone().to(device, non_blocking=True)
        
        #获取超像素中心坐标点、超像素分割结构每个区域的像素个数、超像素总数目
        sup_centre,sup_pix,sup_num=E.my_slic(segment=segment)
        #获取模型输出
        pred_y,Gap_y,all_F,X_F=model(X,img_lab,seg_lab)
        # print('check!')
        # for name, param in model.named_parameters():
        #     if torch.isnan(param).any() or torch.isinf(param).any():
        #         print(name)
        #计算损失函数
        l=loss(sup_num,X_F,segment,all_F,pred_y,Gap_y,c,seg_lab,args).sum()
        #计算预测准确率，后续换成计算评价指标
        loss_train=acc1.update(l)
        l.backward()
        optim.step()
        
        # print("###### end optime step")
        optim.zero_grad()
        if torch.isnan(l):
            print(torch.isnan(torch.sum(torch.log(pred_y))))
            print(torch.isnan(torch.sum(all_F)))
            print(torch.isnan(torch.sum(Gap_y)))
            print("Nan")
            return -1
        tbar.set_description(f'Train_Epoch [{epoch}/{args.epochs}]')
        tbar.set_postfix(loss='{0:1.3f}'.format(loss_train.item()))
    writer.add_scalar('Loss/Train_Avg_Loss', loss_train.item(), epoch)
    return 0        

def train_vu(model_vit,model_unet,dataloader,loss_vit,loss_unet,optim_vit,optim_unet,device,epoch,writer,args):
    model_vit.train()
    model_unet.train()
    tbar=tqdm(dataloader)
    acc_vit=AverageMeter()
    acc_unet=AverageMeter()
    for i, ( X,X_unet,y) in enumerate(tbar):
        y=torch.unsqueeze(y,1)
        # X_clone=X.clone().to(device, non_blocking=True)
        # segment=torch.squeeze(y).numpy()
        # print(y.shape)
        #print(X.dtype)
        X, y = X.clone().to(device, non_blocking=True), y.clone().to(device, non_blocking=True)
        X_unet=X_unet.clone().to(device, non_blocking=True)
        #vit step1
        pred_th=model_vit(X)
        # cal vit loss
        #计算损失函数
        l_vit=loss_vit(pred_th,y).sum()
        #计算预测准确率，后续换成计算评价指标
        loss_vit_train=acc_vit.update(l_vit)
        l_vit.backward()
        optim_vit.step()
        if torch.isnan(l_vit):
            print("vit loss find Nan")
            return -1
        # print("###### end optime step")
        optim_vit.zero_grad()
        #unet step2
        f_map=model_unet(X_unet)
        # result,f_map=model_unet(X.detach())
        # cal unet loss
        l_unet=loss_unet(X_unet[:,0:1,:,:],f_map,pred_th.detach(),args).sum()
        loss_unet_train=acc_unet.update(l_unet) 
        l_unet.backward()
        optim_unet.step()
        optim_unet.zero_grad()
        if torch.isnan(l_unet):
            print("unet loss find Nan")
            return -1
        tbar.set_description(f'Train_Epoch [{epoch}/{args.epochs}]'
                            f"vit_loss: {loss_vit_train.item():.6f} unet_loss: {loss_unet_train.item():.8f} "
                        )
    writer.add_scalar('Loss/Train_Avg_Loss', loss_vit_train.item(), epoch)
    writer.add_scalar('Loss/Train_Avg_Loss', loss_unet_train.item(), epoch)
    return 0
def train_only_unet(model_unet,dataloader,loss_unet,optim_unet,device,epoch,writer,args):
    model_unet.train()
    tbar=tqdm(dataloader)
    acc_vit=AverageMeter()
    acc_unet=AverageMeter()
    for i, ( X,X_unet,y) in enumerate(tbar):
        y=torch.unsqueeze(y,1)
        
        X, y = X.clone().to(device, non_blocking=True), y.clone().to(device, non_blocking=True)
        X_unet=X_unet.clone().to(device, non_blocking=True)
        
        #unet step2
        f_map=model_unet(X_unet)
        # result,f_map=model_unet(X.detach())
        # cal unet loss
        l_unet=loss_unet(X_unet[:,0:1,:,:],f_map,y,args).sum()
        loss_unet_train=acc_unet.update(l_unet) 
        l_unet.backward()
        optim_unet.step()
        optim_unet.zero_grad()
        if torch.isnan(l_unet):
            print("unet loss find Nan")
            return -1
        tbar.set_description(f'Train_Epoch [{epoch}/{args.epochs}]'
                            f"unet_loss: {loss_unet_train.item():.8f} "
                        )
    writer.add_scalar('Loss/Train_Avg_Loss', loss_unet_train.item(), epoch)
    return 0  
def train_only_vit(model_vit,dataloader,loss_vit,loss_unet,optim_vit,device,epoch,writer,args):
    model_vit.train()
    tbar=tqdm(dataloader)
    acc_vit=AverageMeter()
    acc_unet=AverageMeter()
    for i, ( X,X_unet,y) in enumerate(tbar):
        y=torch.unsqueeze(y,1)
        X, y = X.clone().to(device, non_blocking=True), y.clone().to(device, non_blocking=True)
        X_unet=X_unet.clone().to(device, non_blocking=True)
        #vit step1
        pred_th=model_vit(X)
        # cal vit loss
        #计算损失函数
        l_vit=loss_vit(pred_th,y).sum()
        #计算预测准确率，后续换成计算评价指标
        loss_vit_train=acc_vit.update(l_vit)
        l_vit.backward()
        optim_vit.step()
        if torch.isnan(l_vit):
            print("vit loss find Nan")
            return -1
        # print("###### end optime step")
        optim_vit.zero_grad()
        tbar.set_description(f'Train_Epoch [{epoch}/{args.epochs}]'
                            f"vit_loss: {loss_vit_train.item():.6f}  "
                        )
    writer.add_scalar('Loss/Train_Avg_Loss', loss_vit_train.item(), epoch)
    return 0
def test(model,dataloader,device,epoch,loss,writer,args):
    model.eval()
    tbar=tqdm(dataloader)
    acc1=AverageMeter()
    E=slic.SLIC()
    c=torch.ones([1,1])
    c[0,0]=args.c
    c=c.cuda(device=args.device)
    # m=metric.Metric()
    Fb=0.0
    with torch.no_grad():
        for i, ( X, y,img_lab,seg_lab) in enumerate(tbar):
            segment=torch.squeeze(y).numpy()
            X, y,img_lab = X.clone().cuda(device, non_blocking=True), y.clone().cuda(device, non_blocking=True),img_lab.clone().cuda(device, non_blocking=True)
            #获取模型输出
            pred_y,Gap_y,feature_map=model(X)
            sup_centre,sup_pix,sup_num=E.my_slic(segment=segment)
            l=loss(sup_num,X,segment,feature_map,sup_pix,pred_y,Gap_y,c,seg_lab,args).sum()
            loss_test=acc1.update(l)
            merge_segment=merge_sup.merge_sup(result_sup=segment,y_pred=pred_y,feature_map=feature_map,category_num=args.c,beta=0.1,seg_lab=seg_lab)
           
            GT_Merge=merge_segment.cpu().numpy()
            # w,h=GT_Merge.shape
            # # print(GT_Merge.shape)
            # GT_Merge=cv2.resize(GT_Merge,(h//5,w//5),interpolation=cv2.INTER_NEAREST) 
            np.save(args.data_save_path+'/'+str(i)+'.npy', GT_Merge)
            segment1=color.label2rgb(GT_Merge)
            # print(segment1.shape)
            writer.add_image('Seg_Result',segment1,i,dataformats ='HWC')
            cv2.imwrite(args.fig_save_path+'/'+str(i)+'.png',segment1*255)
            tbar.set_description(f'Test_Epoch [{epoch}/{args.epochs}]')
            tbar.set_postfix(loss='{0:1.3f}'.format(loss_test.item()))
            # writer.add_scalar('Test_Avg_loss', loss_test.item(), (epoch-1)*len(dataloader)+i)
        writer.add_scalar('Loss/Test_Avg_Loss', loss_test.item(), epoch)
    return loss_test
            
            
def test_v(model,dataloader,device,epoch,loss,writer,args):
    model.eval()
    tbar=tqdm(dataloader)
    acc1=AverageMeter()
    E=slic.SLIC()
    c=torch.ones([1,1])
    c[0,0]=args.c
    c=c.to(device=args.device)
    # m=metric.Metric()
    Fb=0.0
    with torch.no_grad():
        for i, ( X, y,img_lab,seg_lab) in enumerate(tbar):
            segment=torch.squeeze(y).numpy()
            X, y,img_lab = X.clone().to(device, non_blocking=True), y.clone().to(device, non_blocking=True),img_lab.clone().to(device, non_blocking=True)
            #获取模型输出
            pred_y,Gap_y,all_F,X_F=model(X,img_lab,seg_lab)
            sup_centre,sup_pix,sup_num=E.my_slic(segment=segment)
            l=loss(sup_num,X_F,segment,all_F,pred_y,Gap_y,c,seg_lab,args).sum()
            loss_test=acc1.update(l)
            merge_segment=merge_sup.merge_sup(result_sup=segment,y_pred=pred_y,all_F=all_F,category_num=args.c,beta=0.3,seg_lab=seg_lab)
           
            GT_Merge=merge_segment.cpu().numpy()
            # w,h=GT_Merge.shape
            # # print(GT_Merge.shape)
            # GT_Merge=cv2.resize(GT_Merge,(h//5,w//5),interpolation=cv2.INTER_NEAREST) 
            np.save(args.data_save_path+'/'+str(i)+'.npy', GT_Merge)
            segment1=color.label2rgb(GT_Merge)
            # print(segment1.shape)
            writer.add_image('Seg_Result',segment1,i,dataformats ='HWC')
            cv2.imwrite(args.fig_save_path+'/'+str(i)+'.png',segment1*255)
            tbar.set_description(f'Test_Epoch [{epoch}/{args.epochs}]')
            tbar.set_postfix(loss='{0:1.3f}'.format(loss_test.item()))
            # writer.add_scalar('Test_Avg_loss', loss_test.item(), (epoch-1)*len(dataloader)+i)
        writer.add_scalar('Loss/Test_Avg_Loss', loss_test.item(), epoch)
    return loss_test            

def test_vu(model_vit,model_unet,dataloader,loss_vit,loss_unet,device,epoch,writer,args):
    model_unet.eval()
    model_vit.eval()
    tbar=tqdm(dataloader)
    acc_vit=AverageMeter()
    acc_unet=AverageMeter()

    with torch.no_grad():
        for i, ( X,X_unet, y) in enumerate(tbar):
            y=torch.unsqueeze(y,1)
            # X_clone=X.clone().to(device, non_blocking=True)
            X =X.clone().to(device, non_blocking=True)
            y=y.clone().to(device, non_blocking=True)
            X_unet=X_unet.clone().to(device, non_blocking=True)
            #获取模型输出
            # print(X[0,0].max())
            # print(X[0,0].min())
            # print(X[0,0].mean())    
            # print(y)
            pred_th=model_vit(X)
            f_map=model_unet(X_unet)

            # result,f_map=model_unet(X)
            # print(pred_th)
            #计算损失函数
            l_vit=loss_vit(pred_th,y).sum()
            loss_vit_test=acc_vit.update(l_vit)
            l_unet=loss_unet(X_unet[:,0:1,:,:],f_map,pred_th,args).sum()
            loss_unet_test=acc_unet.update(l_unet)
            B,C,W,H=f_map.shape
            # print(f_map.shape)
            c1=((f_map.permute(0,2,3,1)[:,0,0,:]+f_map.permute(0,2,3,1)[:,127,127,:]+f_map.permute(0,2,3,1)[:,127,0,:]+f_map.permute(0,2,3,1)[:,0,127,:])/4).cpu().numpy()
            c2=((f_map.permute(0,2,3,1)[:,63,63,:]+f_map.permute(0,2,3,1)[:,63,64,:]+f_map.permute(0,2,3,1)[:,64,63,:]+f_map.permute(0,2,3,1)[:,64,64,:])/4).cpu().numpy()
            f_map=f_map.view(B,C,-1).permute(0,2,1)
            f_map=f_map.cpu().numpy()

            # result=result.cpu().numpy()
            # X=X.cpu().numpy()
            result=np.zeros((f_map.shape[0],f_map.shape[1]))
            for j in range(B):
                k_means=KMeans(n_clusters=2, random_state=10,init=np.array([c1[j],c2[j]]))
                result[j]=k_means.fit_predict(f_map[j])
                # clus=AgglomerativeClustering(n_clusters=2,linkage='complete',metric='cosine')
                # result[j]=clus.fit_predict(f_map[j])
                # result[j], error, nfound = kcluster(f_map[j],dist='u',initialid=[c1,c2])

            # for j in range(result.shape[0]):    
            #     result[j]=np.where(X[j,0].cpu().numpy()>pred_th[j].cpu().numpy(),255,0) 
            # result[result>pred_th]=255
            # result[result<=pred_th]=0
            result=result.reshape(B,1,W,H)
            b,c,w,h=result.shape
            # segment1=color.label2rgb(result)
            
            np.save(args.data_save_path+'/'+str(i)+'.npy', result)
            
            writer.add_image('Seg_Result',result,i,dataformats ='NCHW')
            for j in range (b):
                cv2.imwrite(args.fig_save_path+'/'+str(i*b+j)+'.png',result[j,0,:,:]*255)
            tbar.set_description(f'Test_Epoch [{epoch}/{args.epochs}]'
                            f"vit_loss: {loss_vit_test.item():.6f} unet_loss: {loss_unet_test.item():.8f} "
                        )
            # writer.add_scalar('Test_Avg_loss', loss_test.item(), (epoch-1)*len(dataloader)+i)
        writer.add_scalar('Loss/Test_Avg_Loss', loss_vit_test.item(), epoch)
        writer.add_scalar('Loss/Test_Avg_Loss', loss_unet_test.item(), epoch)
    return loss_vit_test,loss_unet_test          

def test_only_unet(model_unet,dataloader,loss_unet,device,epoch,writer,args):
    model_unet.eval()
    tbar=tqdm(dataloader)
    acc_vit=AverageMeter()
    acc_unet=AverageMeter()

    with torch.no_grad():
        for i, ( X,X_unet, y) in enumerate(tbar):
            y=torch.unsqueeze(y,1)
            # X_clone=X.clone().to(device, non_blocking=True)
            X =X.clone().to(device, non_blocking=True)
            y=y.clone().to(device, non_blocking=True)
            X_unet=X_unet.clone().to(device, non_blocking=True)
            #获取模型输出
            f_map=model_unet(X_unet)
            #计算损失函数
            l_unet=loss_unet(X_unet[:,0:1,:,:],f_map,y,args).sum()
            loss_unet_test=acc_unet.update(l_unet)
            B,C,W,H=f_map.shape
            # print(f_map.shape)
            f_map=f_map.view(B,C,-1).permute(0,2,1)
            f_map=f_map.cpu().numpy()
            # result=result.cpu().numpy()
            # X=X.cpu().numpy()
            result=np.zeros((f_map.shape[0],f_map.shape[1]))
            for j in range(B):
                # clus=AgglomerativeClustering(n_clusters=2,linkage='complete',metric='cosine')
                # result[j]=clus.fit_predict(f_map[j])
                result[j], error, nfound = kcluster(f_map[j],dist='u',npass=10)
            result=result.reshape(B,1,W,H)
            b,c,w,h=result.shape
            
            np.save(args.data_save_path+'/'+str(i)+'.npy', result)
            
            writer.add_image('Seg_Result',result,i,dataformats ='NCHW')
            for j in range (b):
                cv2.imwrite(args.fig_save_path+'/'+str(i*b+j)+'.png',result[j,0,:,:]*255)
            tbar.set_description(f'Test_Epoch [{epoch}/{args.epochs}]'
                            f"unet_loss: {loss_unet_test.item():.8f} "
                        )
            # writer.add_scalar('Test_Avg_loss', loss_test.item(), (epoch-1)*len(dataloader)+i)
        writer.add_scalar('Loss/Test_Avg_Loss', loss_unet_test.item(), epoch)
    return loss_unet_test


def test_only_vit(model_vit,dataloader,loss_vit,device,epoch,writer,args):
    model_vit.eval()
    tbar=tqdm(dataloader)
    acc_vit=AverageMeter()
    acc_unet=AverageMeter()

    with torch.no_grad():
        for i, ( X,X_unet, y) in enumerate(tbar):
            y=torch.unsqueeze(y,1)
            X =X.clone().to(device, non_blocking=True)
            B,C,W,H=X.shape
            y=y.clone().to(device, non_blocking=True)
            X_unet=X_unet.clone().to(device, non_blocking=True)
            #获取模型输出
            pred_th=model_vit(X)
            #计算损失函数
            l_vit=loss_vit(pred_th,y).sum()
            loss_vit_test=acc_vit.update(l_vit)
            result=torch.zeros((B,1,W,H))
            for j in range(X_unet.shape[0]):
                result[j]=torch.where(X_unet[j,0]>pred_th[j],torch.ones_like(result[j]),torch.zeros_like(result[j]))

            result=result.reshape(B,1,W,H)
            b,c,w,h=result.shape
            # segment1=color.label2rgb(result)
            
            np.save(args.data_save_path+'/'+str(i)+'.npy', result)
            
            writer.add_image('Seg_Result',result,i,dataformats ='NCHW')
            for j in range (b):
                cv2.imwrite(args.fig_save_path+'/'+str(i*b+j)+'.png',result[j,0,:,:]*255)
            tbar.set_description(f'Test_Epoch [{epoch}/{args.epochs}]'
                            f"vit_loss: {loss_vit_test.item():.6f} "
                        )
            # writer.add_scalar('Test_Avg_loss', loss_test.item(), (epoch-1)*len(dataloader)+i)
        writer.add_scalar('Loss/Test_Avg_Loss', loss_vit_test.item(), epoch)
    return loss_vit_test