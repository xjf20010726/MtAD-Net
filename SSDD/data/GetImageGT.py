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
root_path=os.getcwd()
annotations_path = os.path.join(root_path, "A")
images_path=os.path.join(root_path,"images")
masks_path=os.path.join(root_path,"masks")
targets_img_path=os.path.join(root_path,"Targets")
targets_mask_path=os.path.join(root_path,"Targets_masks")
print(targets_img_path,targets_mask_path)
def imadjust(img,low_in, high_in, low_out, high_out, gamma, c):
    w,h=img.shape
    new_img=np.zeros([w, h])
    for x in range(0, w):
        for y in range(0, h):
            if img[x, y] <= low_in:
                new_img[x, y] = low_out
            elif img[x, y] >= high_in:
                new_img[x, y] = high_out
            else:
                new_img[x, y] = c * (img[x, y]**gamma)
    return new_img
def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args:
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}


def Get_xml(annotation_path='Annotations_Box/',image_path='JPEGImages/',mask_path='JPEGImages_PSeg_GT_instance/',save_img='target_img/',save_mask='target_mask/'):
    if not os.path.isdir(save_img):
        os.makedirs(save_img)

    if not os.path.isdir(save_mask):
        os.makedirs(save_mask)

    imagelist = os.listdir(annotation_path)
    for image in imagelist:  
        image_pre, ext = os.path.splitext(image)
        imgfile = image_path + '/' + image_pre + '.jpg'
        xmlfile = annotation_path + '/'  + image_pre + '.xml'
        maskfile=mask_path+'/'+image_pre+'.jpg'
        # print(image)
        img = cv2.imread(imgfile,1)
        mask=cv2.imread(maskfile,0)
        with open(xmlfile,encoding='utf-8') as fid:
            xml_str = fid.read()
            xml = etree.fromstring(xml_str)
            data = parse_xml_to_dict(xml)["annotation"]
        if "object" in data:
            count=0
            for obj in data["object"]:
                # print(obj)
                # break
                if obj["name"]!="ship":
                    print(obj["name"])
                    continue
                # print(obj)
                # xmin=int(obj["bndbox"]["xmin"])
                # xmax=int(obj["bndbox"]["xmax"])
                # ymin=int(obj["bndbox"]["ymin"])
                # ymax=int(obj["bndbox"]["ymax"])
                xmin = max(0,int(obj["bndbox"]["xmin"])-15)
                xmax = min(int(obj["bndbox"]["xmax"])+15,img.shape[1])
                ymin = max(0,int(obj["bndbox"]["ymin"])-15)
                ymax = min(int(obj["bndbox"]["ymax"])+15,img.shape[0])

                xmin_1=int(obj["bndbox"]["xmin"])
                xmax_1=int(obj["bndbox"]["xmax"])
                ymin_1=int(obj["bndbox"]["ymin"])
                ymax_1=int(obj["bndbox"]["ymax"])
                tmp_img=copy.deepcopy(img[ymin:ymax,xmin:xmax])
                if ymin==0 and int(obj["bndbox"]["ymin"])-15<0:
                    tmp_img=np.pad(tmp_img,((15-int(obj["bndbox"]["ymin"]),0),(0,0),(0,0)),mode='linear_ramp')
                
                if ymax==img.shape[0] and int(obj["bndbox"]["ymax"])+15>img.shape[0]:
                    # print('------')
                    # print(img.shape[0]-int(obj["bndbox"]["ymax"]))
                    # print(10-img.shape[0]+int(obj["bndbox"]["ymax"]))
                    tmp_img=np.pad(tmp_img,((0,int(obj["bndbox"]["ymax"])+15-img.shape[0]),(0,0),(0,0)),mode='linear_ramp')
                
                if xmin==0 and int(obj["bndbox"]["xmin"])-15<0:
                    tmp_img=np.pad(tmp_img,((0,0),(15-int(obj["bndbox"]["xmin"]),0),(0,0)),mode='linear_ramp')
                
                if xmax==img.shape[1] and int(obj["bndbox"]["xmax"])+15>=img.shape[1]:
                    tmp_img=np.pad(tmp_img,((0,0),(0,int(obj["bndbox"]["xmax"])+15-img.shape[1]),(0,0)),mode='linear_ramp')
                # hsv=cv2.cvtColor(tmp_img,cv2.COLOR_BGR2HSV)
                # H, S, V = cv2.split(hsv)
                # # retVal, _ = cv2.threshold(V, 0, 255, cv2.THRESH_OTSU)
                # # V[V>=retVal]=255
                # gamma=1.5
                # new_V=np.power((V/255.0),gamma)*255.0
                
                # new_V=np.clip(new_V, 0, 255)
                # new_V=new_V.astype(np.uint8)
                # new_V=cv2.blur(new_V,(3,3))
                # # new_V=cv2.medianBlur(new_V,3)
                # new_hsv=cv2.merge([H,S,new_V])
                # tmp_img=cv2.cvtColor(new_hsv,cv2.COLOR_HSV2BGR)
                # V=V.astype(np.float32)
                # V/=255.0
                # new_V=imadjust(V,0.1,0.7,0,1,1.5,1)
                # new_V=(new_V*255).astype(np.uint8)
                # new_hsv=cv2.merge([H,S,new_V])
                # tmp_img=cv2.cvtColor(new_hsv,cv2.COLOR_HSV2BGR)
                tmp_mask=copy.deepcopy(mask[ymin_1:ymax_1,xmin_1:xmax_1])    
                tmp_mask=np.pad(tmp_mask,((15,15),(15,15)),mode='constant',constant_values=0)
                # tmp_img=copy.deepcopy(img[ymin_1:ymax_1,xmin_1:xmax_1])
                if image=='000399.xml':
                    print(tmp_img.shape)
                    print(tmp_mask.shape)
                if tmp_img.shape[0]-20!=int(obj["bndbox"]["ymax"])-int(obj["bndbox"]["ymin"]) or tmp_img.shape[1]-20!=int(obj["bndbox"]["xmax"])-int(obj["bndbox"]["xmin"]):
                    
                    print(image,tmp_img.shape,(int(obj["bndbox"]["ymax"])-int(obj["bndbox"]["ymin"]),int(obj["bndbox"]["xmax"])-int(obj["bndbox"]["xmin"])))
                
                count+=1
                target_path="%sTarget%02d.png"%(image_pre,count)
                # print(target_path)
                # save_t1_path=os.path.join("Targets",target_path)
                cv2.imwrite(save_img+target_path,tmp_img)
                cv2.imwrite(save_mask+target_path,tmp_mask)



# xml_names=os.listdir(annotations_path)
# print(xml_names)
# for xml_name in xml_names:
    
#     xml_path=os.path.join(annotations_path,xml_name)

#     #print(xml_path)

#     if os.path.exists(xml_path) is False:
#         print(f"Warning: not found '{xml_path}', skip this annotation file.")
#     with open(xml_path,encoding='utf-8') as fid:
#         xml_str = fid.read()
#         xml = etree.fromstring(xml_str)
#         data = parse_xml_to_dict(xml)["annotation"]
        
        
#     image_name=data["filename"]
#     print(image_name)
#     img_path=os.path.join("images",data["filename"])
#     print(img_path)
#     image=cv2.imread(img_path,flags=1)
# #     pauli_hsv=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
# #     L, A, B = cv2.split(pauli_hsv)
# #     print(L.max())
# #     gamma=4
# #     new_L=np.power((L/255.0),gamma)*255.0
    
# #     new_L=np.clip(new_L, 0, 255)
# #     new_L=new_L.astype(np.uint8)
# #     new_lab=cv2.merge([new_L,A,B])
# #     image=cv2.cvtColor(new_lab,cv2.COLOR_LAB2BGR)
# #     image=cv2.medianBlur(image,3)
# #     min_val=np.percentile(L.flatten(),90)
# #     max_val=np.percentile(L.flatten(),95)
# #     new_V=imadjust(V,min_val,max_val,50,255,1,1)
# #     print(L.dtype)
# #     print(new_V.dtype)
# #     new_hsv=cv2.merge([new_V,A,B])
# #     image=cv2.cvtColor(new_hsv,cv2.COLOR_LAB2BGR)
# #     image=cv2.medianBlur(image,7)
# #     cv2.imwrite("images/new_filter_"+data["filename"],image)
#     if "object" in data:
#         count=0
#         for obj in data["object"]:
            
#             xmin = int(obj["bndbox"]["xmin"])
#             xmax = int(obj["bndbox"]["xmax"])
#             ymin = int(obj["bndbox"]["ymin"])
#             ymax = int(obj["bndbox"]["ymax"])
            
#             tmp_img=copy.deepcopy(image[ymin:ymax,xmin:xmax])
#             pauli_lab=cv2.cvtColor(tmp_img,cv2.COLOR_BGR2LAB)
#             L, A, B = cv2.split(pauli_lab)
#             retVal, _ = cv2.threshold(L, 0, 255, cv2.THRESH_OTSU)
#             L[L>=retVal]=255
# #             gamma=1.1
# #             new_L=np.power((L/255.0),gamma)*255.0

# #             new_L=np.clip(new_L, 0, 255)
#             new_L=L.astype(np.uint8)
#             new_lab=cv2.merge([new_L,A,B])
#             tmp_img=cv2.cvtColor(new_lab,cv2.COLOR_LAB2BGR)
#             tmp_img=cv2.resize(tmp_img,(128,128))
#             count+=1
#             target_path="%sTarget%02d.png"%(image_name[:-4],count)
#             save_t1_path=os.path.join("Targets",target_path)
# #             save_t_path=os.path.join("Targets_masks",target_path)
#             cv2.imwrite(save_t1_path,tmp_img)
# #             cv2.imwrite(save_t_path,tmp_img1)
Get_xml(annotation_path='Annotations_train/',save_img='train/img/',save_mask='train/mask/')
Get_xml(annotation_path='Annotations_test/',save_img='test/img/',save_mask='test/mask/')