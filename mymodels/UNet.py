import torch
import torch.nn as nn
import torch.nn.functional as F
from mymodels.conv_utils import DownConv,UpConv
    



class UNet(nn.Module):
    def __init__(self, in_channel, channel_nums,out_channel) -> None:
        super().__init__()
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
        # self.up1=UpConv(channel_nums[4]+channel_nums[4],channel_nums[4])
        #up2=[B,channel_nums[4],M/16,N/16] concat [B,channel_nums[3],M/8,N/8]
        self.up2=UpConv(channel_nums[3]+channel_nums[3],channel_nums[3])
        #up3= [B,channel_nums[3],M/8,N/8] concat [B,channel_nums[2],M/4,N/4]
        self.up3=UpConv(channel_nums[3]+channel_nums[2],channel_nums[2])
        #up4 = [B,channel_nums[2],M/4,N/4] [B,channel_nums[1],M/2,N/2]
        self.up4=UpConv(channel_nums[2]+channel_nums[1],channel_nums[1])
        #up5 = [B,channel_nums[1],M/2,N/2] [B,channel_nums[0],M,N]
        self.up5=UpConv(channel_nums[1]+channel_nums[0],channel_nums[0])
        self.conv = nn.Conv2d(channel_nums[0], channel_nums[0], 1)
        self.soft=nn.Sequential(
            nn.Conv2d(in_channels=channel_nums[0],out_channels=out_channel,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=out_channel),
            nn.Sigmoid()
        )
        self.GAP=nn.AdaptiveAvgPool2d(1)

    
    def forward(self,x):
        f1,p1,i1=self.down1(x)
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
        conv_1=F.normalize(conv, p=2, dim=1, eps=1e-12)
        result=self.soft(conv_1)

        return result,self.GAP(result),conv_1




if __name__=="__main__":
    conv1=DownConv(3,64)
    x = torch.randn(1, 3, 224, 224)
    conv_1,pool_1,_=conv1(x)

    print(conv_1.shape)
    print(pool_1.shape)

    conv2=UpConv(64+64,3)

    conv_2=conv2(pool_1,conv_1,_)
    print(conv_2.shape)

    net=UNet(3,[16,32,64,128,256],32)
    result,Gap,u5=net(x)
    print(result.shape)
    print(Gap.shape)
    print(u5.shape)
