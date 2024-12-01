# -*- coding: utf-8 -*-
from __future__ import division 
import os
import xml.dom.minidom
import cv2
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from shapely.geometry import Polygon

font = {'family' : 'Palatino Linotype',
'weight' : 'normal',
'size'   : 15,
}

ImgPath = 'JPEGImages_test/'
AnnoPath = 'Annotations_test/'

imagelist = os.listdir(AnnoPath)
width_list = []
height_list = []

objects_list = []

area_ratio_list = []



for image in imagelist:
    image_pre, ext = os.path.splitext(image)
    imgfile = ImgPath + '/' + image_pre + '.jpg'
    xmlfile = AnnoPath + '/'  + image_pre + '.xml'
    
    im = cv2.imread(imgfile)
    
    DomTree = xml.dom.minidom.parse(xmlfile)
    annotation = DomTree.documentElement
    
    filename = annotation.getElementsByTagName('filename')[0].childNodes[0].data
    
    size = annotation.getElementsByTagName('size')[0]
    
    # print('size =', size)
    
    width = int(size.getElementsByTagName('width')[0].childNodes[0].data)
    height = int(size.getElementsByTagName('height')[0].childNodes[0].data)
    
    width_list.append(width)
    height_list.append(height)
    
    objectlist = annotation.getElementsByTagName('object')
    
    for i, objects in enumerate(objectlist):
        namelist = objects.getElementsByTagName('name')
        objectname = namelist[0].childNodes[0].data
        # points = objects.getElementsByTagName('segm')[0]
        segm = objects.getElementsByTagName('segm')
        
        points_list = []
        
        for one_segm in segm:
            
            points = one_segm.getElementsByTagName('point')
            
            for point in points:
                
                # print(point.getElementsByTagName('point')[0].childNodes[0].data)
                
                x = int(point.childNodes[0].data.split(',')[0])
                y = int(point.childNodes[0].data.split(',')[1])
                
                points_list.append((x, y))
            
        objects_list.append(points_list)
        
        
        area_ratio = (Polygon(points_list).convex_hull.area / (width * height)) * 100
        area_ratio_list.append(area_ratio)
# print(points_list)

area_list = []
perimeter_list = []

for one_seg in objects_list:

    area = Polygon(one_seg).convex_hull.area
    perimeter = Polygon(one_seg).convex_hull.length
    
    area_list.append(area)
    perimeter_list.append(perimeter)


# ------------------------------------------------------------------------------------
plt.figure(1)
area_list=np.array(area_list)/10000
arr=plt.hist(area_list, bins=40, color='red', edgecolor='w')

for i in range(40):
    plt.text(arr[1][i],arr[0][i], str(int(arr[0][i])), fontsize=6)
    
plt.xlim(0, 27500/10000)
# plt.xlim(0, 27500)
plt.ylim(0, 1750)

plt.xlabel('PSeg Ship Area ' + "(×${10}^{4}$)", font)
plt.ylabel('Number of Images', font)
plt.savefig('Area_PSeg_Ship.png', bbox_inches='tight', dpi=1200)
plt.show()


# ------------------------------------------------------------------------------------
plt.figure(2)

arr=plt.hist(perimeter_list, bins=20, color='blue', edgecolor='w')

for i in range(20):
    plt.text(arr[1][i],arr[0][i], str(int(arr[0][i])))
    
plt.xlim(0, 800)
plt.ylim(0, 800)

plt.xlabel('PSeg Ship Perimeter ', font)
plt.ylabel('Number of Images', font)
plt.savefig('PSeg_Ship_Perimeter.png', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------------
plt.figure(3)

arr=plt.hist(area_ratio_list, bins=20, color='coral', edgecolor='w', label='%')

for i in range(20):
    plt.text(arr[1][i],arr[0][i], str(int(arr[0][i])), fontsize=6)
    
plt.xlim(0, 15)
plt.ylim(0, 2100)

plt.xlabel('Area Ratio of PSeg Ship among the Whole Image (%)', font)
plt.ylabel('Number of Images', font)
plt.savefig('Area Ratio of PSeg Ship among the Whole Image.png', bbox_inches='tight', dpi=1200)
plt.show()
