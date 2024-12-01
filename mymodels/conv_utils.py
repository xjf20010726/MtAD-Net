import torch
import torch.nn as nn
import torch.nn.functional as F

class DownConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv_1=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_2=nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            
        )
        self.max_pool=nn.MaxPool2d(kernel_size=2,return_indices=True,stride=2)
        self.shortcut=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,stride=1),
            nn.InstanceNorm2d(num_features=out_channels),
        )
    def forward(self,x):
        # print(x.shape)
        conv_1=self.conv_1(x)
        # print(conv_1.shape)
        # print(self.shortcut(x).shape)
        conv_2=self.conv_2(conv_1)+self.shortcut(x)
        # print(conv_2.shape)
        pool_1, indices_1=self.max_pool(conv_2)
        return conv_2,pool_1,indices_1
    

class UpConv(nn.Module):
    def __init__(self, in_channel_x1,in_channel_x2,out_channels):
        super().__init__()
        self.max_unpool=nn.MaxUnpool2d(2,stride=2)
        self.up=nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # self.up=nn.ConvTranspose2d(in_channel_x1,in_channel_x1,kernel_size=3,stride=2,padding=1)
        self.conv_1=nn.Sequential(
            nn.Conv2d(in_channels=in_channel_x1+in_channel_x2,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),            
        )
        self.conv_2=nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,stride=1,padding=1),
            nn.InstanceNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),            
        )
        self.shortcut=nn.Sequential(
            nn.Conv2d(in_channels=in_channel_x1+in_channel_x2,out_channels=out_channels,kernel_size=1,stride=1),
            nn.InstanceNorm2d(num_features=out_channels),
        )
    def forward(self,x1,x2,indices):
        # max_unpool=self.max_unpool(x1,indices)

        x1=self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        #print(up.shape)
        #print(x2.shape)
        x=torch.cat([x2,x1],dim=1)
        conv=self.conv_1(x)
        conv=self.conv_2(conv)+self.shortcut(x)
        return conv