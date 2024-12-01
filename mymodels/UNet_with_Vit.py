import torch
import torch.nn as nn
import torch.nn.functional as F
from mymodels.conv_utils import DownConv,UpConv
from mymodels.Vit import VisionTransformer 
import numpy as np
# from conv_utils import DownConv,UpConv
# from Vit import VisionTransformer 
class UNet_Vit(nn.Module):
    def __init__(self, in_channel, channel_nums,feature_channel=32,out_channel=1,patch_dim=128,embed_dim=256,feature_dim=32, depth=12, num_heads=8):
        super().__init__()
        self.feature_channel=feature_channel
        self.vit=VisionTransformer(patch_dim=patch_dim,embed_dim=embed_dim,feature_dim=feature_dim,depth=depth, num_heads=num_heads)
        #[16,32,64,128,256]
        #[B,in_channel,M,N]->[B,channel_nums[0],M,N],[B,channel_nums[0],M/2,N/2]
        self.down1=DownConv(in_channel,channel_nums[0])
        #[B,channel_nums[0],M/2,N/2]->[B,channel_nums[1],M/2,N/2],[B,channel_nums[1],M/4,N/4]
        self.down2=DownConv(channel_nums[0],channel_nums[1])
        #[B,channel_nums[1],M/4,N/4]->[B,channel_nums[2],M/4,N/4],[B,channel_nums[2],M/8,N/8]
        self.down3=DownConv(channel_nums[1],channel_nums[2])
        #[B,channel_nums[2],M/8,N/8]->[B,channel_nums[3],M/8,N/8],[B,channel_nums[3],M/16,N/16]
        self.down4=DownConv(channel_nums[2],channel_nums[3])
        #[B,channel_nums[3],M/16,N/16]->[B,channel_nums[4],M/16,N/16],[B,channel_nums[4],M/32,N/32]
        # self.down5=DownConv(channel_nums[3],channel_nums[4])

        #up1=[B,channel_nums[4],M/32,N/32] concat [B,channel_nums[4],M/16,N/16]
        # self.up1=UpConv(channel_nums[4]+channel_nums[4],feature_channel)
        #up2=[B,channel_nums[4],M/16,N/16] concat [B,channel_nums[3],M/8,N/8]
        self.up2=UpConv(channel_nums[3]+channel_nums[3],feature_channel)
        #up3= [B,channel_nums[3],M/8,N/8] concat [B,channel_nums[2],M/4,N/4]
        self.up3=UpConv(channel_nums[2]+feature_channel,feature_channel)
        #up4 = [B,channel_nums[2],M/4,N/4] [B,channel_nums[1],M/2,N/2]
        self.up4=UpConv(channel_nums[1]+feature_channel,feature_channel)
        #up5 = [B,channel_nums[1],M/2,N/2] [B,channel_nums[0],M,N]
        self.up5=UpConv(channel_nums[0]+feature_channel,feature_channel)
        self.conv = nn.Conv2d(feature_channel, feature_channel, 1)
        self.soft=nn.Sequential(
            nn.Conv1d(in_channels=feature_channel*2,out_channels=out_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm1d(num_features=out_channel),
            nn.Sigmoid()
        )
        self.GAP=nn.AdaptiveAvgPool1d(1)

        
    
    def forward(self,x,img_lab,seg_lab=None):
        if torch.isnan(img_lab).any():
            print("img_lab has nan")
        _,_,Sup_num,_=img_lab.shape
        vit_feature=self.vit(img_lab)
        if torch.isnan(vit_feature).any():
            print("vit_feature has nan")
        f1,p1,i1=self.down1(x)
        if torch.isnan(f1).any():
            print("f1 has nan")
        if torch.isnan(p1).any():
            print("p1 has nan")
        f2,p2,i2=self.down2(p1)
        f3,p3,i3=self.down3(p2)
        f4,p4,i4=self.down4(p3)
        # f5,p5,i5=self.down5(p4)


        # u1=self.up1(p5,f5,i5)
        #print(u1.shape)
        u2=self.up2(p4,f4,i4)
        u3=self.up3(u2,f3,i3)
        u4=self.up4(u3,f2,i2)
        u5=self.up5(u4,f1,i1)
        conv=self.conv(u5)
        # with torch.no_grad():
        #     Superpixel_num=len(seg_lab)

        #     self.Mean_Cnn_F=torch.zeros(Superpixel_num,self.feature_channel).cuda(device=args.device)
        #     F1=conv.permute(0,2,3,1).view(-1,self.feature_channel)
        #     for i,inds in enumerate(seg_lab):
        #         inds=inds.numpy().squeeze()
        #         # print(inds.shape)
        #         self.Mean_Cnn_F[i,:]=torch.mean(F1[inds,:],dim=0)
            
        conv_1=F.normalize(conv, p=2, dim=1, eps=1e-12)
        # if torch.isnan(conv_1).any():
        #     print("conv_1 has nan")
        # print(conv_1)
        batch_size,channel_f,_,_=conv_1.shape
        F1=conv_1.permute(0,2,3,1).view(-1,channel_f)
        # print(F1.shape)
        M_F=torch.zeros(Sup_num,channel_f).to(device=x.device)
        with torch.no_grad():
            X_F=torch.zeros(Sup_num,x.shape[1]).to(device=x.device)
            X=x.permute(0,2,3,1).view(-1,x.shape[1])
        for i,inds in enumerate(seg_lab):
            inds=inds.numpy().squeeze()
            # inds=inds.squeeze()
            # print(inds.shape)
            M_F[i,:]=torch.mean(F1[inds,:],dim=0)
            with torch.no_grad():
                X_F[i,:]=torch.mean(X[inds,:],dim=0)
            #M_F shape-->[Sup_num,channel_f]
        M_F=M_F.view(batch_size,Sup_num,channel_f)
        with torch.no_grad():
            X_F=X_F.view(batch_size,Sup_num,x.shape[1]).permute(0,2,1)
        vit_feature=F.normalize(vit_feature, p=2, dim=-1, eps=1e-12)
        all_F=torch.concat((M_F,vit_feature),dim=-1).permute(0,2,1)#[B,channel_f*2,Sup_num]
        all_F=F.normalize(all_F, p=2, dim=1, eps=1e-12)
        
        result=self.soft(all_F)
        # print(result)
        return result,self.GAP(result),all_F,X_F


if __name__=="__main__":
    conv1=DownConv(3,64)
    seg_lab=np.load("/home/XJF/code/HRSID/data/SegLab/P0001_0_800_7200_8000Target00.npy",allow_pickle=True).tolist()
    # seg_lab=torch.tensor(seg_lab)
    L=len(seg_lab)
    x = torch.randn(1, 3, 224, 224)
    img_lab=torch.randn(1,1,L,128)
    img_lab2=torch.randn(1,1,L,128)
    # seg_lab=torch.randn(1,118,128)
    conv_1,pool_1,_=conv1(x)

    print(conv_1.shape)
    print(pool_1.shape)

    conv2=UpConv(64+64,3)

    conv_2=conv2(pool_1,conv_1,_)
    print(conv_2.shape)
    x=x.cuda()
    img_lab=img_lab.cuda()
    net=UNet_Vit(3,[16,32,64,128,256],32).cuda()
    result,Gap,f,x_F=net(x,img_lab,seg_lab)
    print(result.shape)
    print(Gap.shape)
    # print(u5.shape)
    print(f.shape)
    print(x_F[0])
    # result,Gap,u5,f=net(x,img_lab2)
    # print(result.shape)
    # print(Gap.shape)
    # print(u5.shape)
    # print(f)