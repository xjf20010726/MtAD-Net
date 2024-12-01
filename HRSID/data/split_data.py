import numpy as np
import os
import random
import shutil


def moveFile(Dir='targets',train_ratio=0.8,val_ratio=0.2):

    # if not os.path.exists(os.path.join(Dir, 'train')):
    #     os.makedirs(os.path.join(Dir, 'train'))
    
    # if not os.path.exists(os.path.join(Dir, 'val')):
    #     os.makedirs(os.path.join(Dir, 'val'))

    filenames = []
    for root,dirs,files in os.walk(Dir):
        for name in files:
            filenames.append(name)
        break
    
    filenum = len(filenames)

    num_train = int(filenum * train_ratio)
    sample_train = random.sample(filenames, num_train)

    for name in sample_train:
        with open("train.txt", "a") as file:
            file.write(name + "\n")
        pass
    #     shutil.move(os.path.join(Dir, name), os.path.join(Dir, 'train'))

    sample_val = list(set(filenames).difference(set(sample_train)))

    for name in sample_val:
        with open("test.txt", "a") as file:
            file.write(name + "\n")
        pass
        # shutil.move(os.path.join(Dir, name), os.path.join(Dir, 'val'))

def getFile(Dir='train/targets/',txt='train.txt'):
    filenames = []
    for root,dirs,files in os.walk(Dir):
        for name in files:
            filenames.append(name)
    
    for name in filenames:
        with open(txt, "a") as file:
            file.write(name + "\n")
    return filenames

getFile()
getFile('test/targets/',txt='test.txt')
# moveFile()