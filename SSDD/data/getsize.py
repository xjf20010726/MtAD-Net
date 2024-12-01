import cv2
import numpy as np
import os
import json
from PIL import Image
from lxml import etree
import copy
import queue
from skimage import io,measure,color
import matplotlib.pyplot as plt
import xml.dom.minidom

def getsize(img_path):
    img = cv2.imread(img_path,0)
    pix_num=np.sum(img==255)
    return pix_num

pix_num1=getsize('/home/XJF/code/HRSID/data/test/mask/P0126_1200_2000_3000_3800Target01.png')
print(pix_num1)
pix_num2=getsize('/home/XJF/code/HRSID/data/test/mask/P0022_1800_2600_7800_8600Target02.png')
print(pix_num2)

pix_num1=getsize('/home/XJF/code/SSDD_Seg/data/test/mask/000031Target01.png')
print(pix_num1)
pix_num2=getsize('/home/XJF/code/SSDD_Seg/data/test/mask/001109Target01.png')
print(pix_num2)