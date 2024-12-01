# -*- coding: utf-8 -*-
from __future__ import division 
import os
import xml.dom.minidom
import cv2
import numpy as np
import random

from shapely.geometry import Polygon

def read_xml(ImgPath = 'JPEGImages/', AnnoPath = 'Annotations/', Savepath = 'JPEGImages_PSeg_GT_instance/'):

    if not os.path.isdir(Savepath):
        os.makedirs(Savepath)

    imagelist = os.listdir(AnnoPath)
    
    points_list = []
    

    for image in imagelist:  
        image_pre, ext = os.path.splitext(image)
        imgfile = ImgPath + '/' + image_pre + '.jpg'
        xmlfile = AnnoPath + '/'  + image_pre + '.xml'
        im = cv2.imread(imgfile)
        k=im.shape
        im_tmp=np.zeros((k[0],k[1]),dtype=np.uint8)
        DomTree = xml.dom.minidom.parse(xmlfile)
        annotation = DomTree.documentElement
        filenamelist = annotation.getElementsByTagName('filename')
        filename = filenamelist[0].childNodes[0].data
        objectlist = annotation.getElementsByTagName('object')
        
        for objects in objectlist:
            namelist = objects.getElementsByTagName('name')
            objectname = namelist[0].childNodes[0].data
            segm = objects.getElementsByTagName('segm')
            
            # for one_segm in segm:
                    
            one_segm_points_list = []
            
            for one_segm in segm:
                
                points = one_segm.getElementsByTagName('point')
                
                for point in points:
                    
                    x = point.childNodes[0].data.split(',')[0]
                    y = point.childNodes[0].data.split(',')[1]
                    
                    one_segm_points_list.append([x, y])
    
            
            
            pts = np.array(one_segm_points_list, np.int32)
            
            

            # cv2.rectangle(im,(xmin,ymin),(xmax,ymax), (0, 255, 0), 2)
            
            r = 255

            g = 255

            b = 255
            
            
            # im = np.zeros([im.shape[0], im.shape[1], 3], np.uint8)
            
            
            print('pts = ', pts)
            
            cv2.fillPoly(im_tmp, [pts], 255)
            
            xmin = np.min(np.array(pts)[:, 0])
            
            ymin = np.min(np.array(pts)[:, 1])
            
            xmax = np.max(np.array(pts)[:, 0])
            
            ymax = np.max(np.array(pts)[:, 1])
            
            print('xmin, ymin, xmax, ymax = ', xmin, ymin, xmax, ymax)
            
            
            # cv2.rectangle(im, (xmin,ymin),(xmax,ymax), (b, g, r), 2)
            
            # cv2.polylines(im,[pts],True,(0,255,0), 2)
        
        path = Savepath + '/' + image_pre + '.jpg'
                    
        cv2.imwrite(path, im_tmp)
read_xml()