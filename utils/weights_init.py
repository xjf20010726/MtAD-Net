import torch
import torch.nn as nn
#对模型参数初始化
class WeightInit(object):
    def __init__(self):
        pass
    #Xavier方法
    def weights_init_xavier(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m=nn.init.xavier_normal_(m.weight.data)
    #kaiming方法
    def weights_init_kaiming(self,m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            m=nn.init.kaiming_normal_(m.weight.data)

    