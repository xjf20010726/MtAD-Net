import torch
import torch.nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
class SaveModel(object):
    def __init__(self) -> None:
        pass
    def save(self,model,path):
        torch.save(model,path)
    def save_checkpoint(self,model,path,epoch,optimizer,lr_scheduler,loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'lr_scheduler_state_dict':lr_scheduler.state_dict(),
        },path)
        pass
class LoadModel():
    def __init__(self) -> None:
        pass
    def load(self,path):
        return torch.load(path,map_location='cuda:1')
    def load_checkpoint(self,path,model,optimizer,lr_scheduler):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model,optimizer,lr_scheduler,epoch,loss

    pass