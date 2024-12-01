import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 
# os.environ["WORLD_SIZE"] = "1"
import sys
sys.path.append('..')
import torchvision.models as models
import torchvision
from torchvision import transforms
from torchvision import datasets
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import time
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from datetime import timedelta
from tqdm             import tqdm
import math

from utils.train_test_utils import *
from utils.weights_init import *
from utils.save_load_model import *
from utils import my_dataset
#from utils import egb 

from loss import Loss
from loss import Loss_New_Model
from loss import MyLoss
from loss import FinetuneLoss
from torchvision.transforms import ToTensor

from utils.earlyStop import EarlyStopping
from torch.utils.tensorboard import SummaryWriter

from mymodels import vit_finetune
from mymodels import unet_finetune
from utils.engine_finetune import train_one_epoch,test_one_epoch
# from mae.util import misc,pos_embed
# from mae import models_mae
# from mae import models_unet

# sys.path.append('/home/XJF/SAR_Ground_Truth/utils')
def main(args):
    # 导入数据集，使用VOC数据集作为训练集，BSDS500作为测试集
    transform = transforms.Compose([
        transforms.Resize([96,96]),
        transforms.ToTensor(),
        #transforms.Resize((512,512))
        # transforms.Normalize([0.432284, 0.4364989, 0.36451283], [ 0.24840787, 0.23247136, 0.24171728]),
        #transforms.Normalize([ 0.1583,  0.1583, 0.1583], [ 0.0885,  0.0885,  0.0885])
        ])
    transform1 = transforms.Compose([
        transforms.ToTensor(),
        ])
    train_thr_dirs=['/home/XJF/code/SSDD_Seg/data/train/threshold_0','/home/XJF/code/SSDD_Seg/data/train/threshold_1',
              '/home/XJF/code/SSDD_Seg/data/train/threshold_2','/home/XJF/code/SSDD_Seg/data/train/threshold_3',
              '/home/XJF/code/SSDD_Seg/data/train/threshold_4']
    training_data=my_dataset.MYDataset(
        annotations_file="/home/XJF/code/SSDD_Seg/data/train.txt",
        img_dir="/home/XJF/code/SSDD_Seg/data/train/img",
        threshold_dirs=train_thr_dirs,
        # index_dir='/home/XJF/SAR_Ground_Truth/data/BSR/BSDS500/SupIndexs',
        transform=transform,
        target_transform=transform1
    )
    test_thr_dirs=['/home/XJF/code/SSDD_Seg/data/test/threshold_0','/home/XJF/code/SSDD_Seg/data/test/threshold_1',
              '/home/XJF/code/SSDD_Seg/data/test/threshold_2','/home/XJF/code/SSDD_Seg/data/test/threshold_3',
              '/home/XJF/code/SSDD_Seg/data/test/threshold_4']


    testing_data=my_dataset.MYDataset(
        annotations_file="/home/XJF/code/SSDD_Seg/data/test.txt",
        img_dir="/home/XJF/code/SSDD_Seg/data/test/img",
        threshold_dirs=test_thr_dirs,
        # index_dir='/home/XJF/SAR_Ground_Truth/data/BSR/BSDS500/SupIndexs',
        transform=transform,
        target_transform=transform1
    )
    train_dataloader = DataLoader(training_data, batch_size=args.batch_size,num_workers=12,shuffle=True)
    test_dataloader = DataLoader(testing_data, batch_size=args.batch_size,num_workers=12,shuffle=False)
    
    Save=SaveModel()
    Load=LoadModel()
    device = args.device
    #导入模型，后续需要换成U-Net

    model_vit=vit_finetune.vit_finetune(num_classes=5)
    model_unet=unet_finetune.unet_finetune()
    checkpoint_vit=torch.load(args.weight_vit,map_location='cpu')
    pretrain_vit = checkpoint_vit['model']
    checkpoint_unet=torch.load(args.weight_unet,map_location='cpu')
    pretrain_unet = checkpoint_unet['model']
    # vit_dict=pretrain_vit.state_dict()
    # unet_dict=pretrain_unet.state_dict()

    model_vit.load_state_dict(pretrain_vit,strict=False)
    model_unet.load_state_dict(pretrain_unet,strict=False)

    model_vit.to(device)
    # print(model_vit.parameters().device)
    model_unet.to(device)
    #定义损失函数和梯度优化，后续换自己的损失函数
    # loss_fn = nn.CrossEntropyLoss()
    #自己的损失函数
    loss_fn=FinetuneLoss.FinetuneLoss()
    optimizer=optim.Adam(params=[{'params':model_vit.parameters()},{'params':model_unet.parameters()}],lr=args.lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs)



    early_stopping_vit = EarlyStopping(args.result+args.save_model_name1)
    early_stopping_unet = EarlyStopping(args.result+args.save_model_name2)
    #确认是否需要载入模型
    if args.if_load:
        #只加载模型参数
        if args.if_checkpoint==0:
            model_vit=Load.load(args.result+args.load_model_name1)
            model_unet=Load.load(args.result+args.load_model_name2)
        #加载所有参数
        else :
            model_vit,optimizer,scheduler,args.start_epoch,loss_fn=Load.load_checkpoint(args.result+'0'+args.load_model_name1,model_vit,optimizer,scheduler)
            model_unet,optimizer,scheduler,args.start_epoch,loss_fn=Load.load_checkpoint(args.result+'0'+args.load_model_name2,model_unet,optimizer,scheduler)
        
    #记录模型训练的时间
    epoch_time=AverageMeter()
    writer = SummaryWriter('runs/SSDD')
    #训练模型
    val=0
    early_stop=0
    for i in range(args.start_epoch,args.epochs):
        start_time=time.time()
        if args.if_test==0:
            devices = []
            sd = model_vit.state_dict()
            for v in sd.values():
                if v.device not in devices:
                    devices.append(v.device)

                
            print(devices)
            val=train_one_epoch(model_vit=model_vit,model_unet=model_unet,dataloader=train_dataloader,loss=loss_fn,optim=optimizer,device=device,epoch=i+1,args=args)
        
        if val==-1:
            break 
        # if (i%16==0) or i==(args.epochs-1):
        #     print("test")
        
        test_loss=test_one_epoch(model_vit=model_vit,model_unet=model_unet,dataloader=test_dataloader,loss=loss_fn,device=device,epoch=i+1,args=args)
        scheduler.step()
        end_time=time.time()-start_time
        epoch_time.update(end_time)
        
        early_stopping_vit(test_loss, model_vit)
        early_stopping_unet(test_loss, model_unet)
        early_stop=(early_stopping_vit.early_stop and early_stopping_unet.early_stop)

        #保存每轮训练的模型，后续需要是否需要保存的条件
        if args.if_checkpoint==1:
            Save.save_checkpoint(model_vit,args.result+str(i)+args.save_model_name1,i,optimizer,scheduler,loss_fn)   
            Save.save_checkpoint(model_unet,args.result+str(i)+args.save_model_name2,i,optimizer,scheduler,loss_fn)
        if early_stop:
            print("Early stopping")
            
            break #跳出迭代，结束训练
    #保存最终模型
    
    Save.save(model_vit,args.result+args.save_model_name1)
    Save.save(model_unet,args.result+args.save_model_name2)
    print(epoch_time.avg)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    
    parser.add_argument('--epochs',
                        default=5,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b',
                        '--batch-size',
                        default=2,
                        type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                        'batch size of all GPUs on the current node when '
                        'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=1e-3,
                        type=float,
                        metavar='LR',
                        help='initial learning rate',
                        dest='lr')
    
    parser.add_argument('--seed', default=1,type=int,help='seed for initializing training. ')
    parser.add_argument('--if_load',default=0,type=int)
    parser.add_argument('--device',default='cuda:0',type=str)
    parser.add_argument('--result',default='/home/XJF/code/SSDD_Seg/results/Model',type=str)
    parser.add_argument('--save_model_name1',default='CNN.pth',type=str)
    parser.add_argument('--save_model_name2',default='CNN.pth',type=str)
    parser.add_argument('--load_model_name1',default='CNN.pth',type=str)
    parser.add_argument('--load_model_name2',default='CNN.pth',type=str)
    parser.add_argument('--if_checkpoint',default=0,type=int)
    parser.add_argument('--fig_save_path',default="/home/XJF/code/SSDD_Seg/results/fig_results_vit_unet_cfar_kmeans",type=str)
    # parser.add_argument('--Fb_save_path',default="/home/XJF/SAR_Ground_Truth/results/Fb_results/slic_result.txt",type=str)
    parser.add_argument('--if_test',default=0,type=int)
    parser.add_argument('--weight_vit',default='/home/XJF/code/mae/output_dir/checkpoint-0.pth',type=str)
    parser.add_argument('--weight_unet',default='/home/XJF/code/mae/output_unet_dir/checkpoint-0.pth',type=str)
    args = parser.parse_args()
    print(args)
    main(args)
