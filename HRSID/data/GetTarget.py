from pycocotools.coco import COCO
from PIL import Image
import os
import tqdm
import cv2
# import imgviz
import numpy as np
import copy
# def save_colored_mask(save_path, mask):
#     """保存调色板彩色图"""
#     lbl_pil = Image.fromarray(mask.astype(np.uint8), mode='P')
#     colormap = imgviz.label_colormap(80)
#     lbl_pil.putpalette(colormap.flatten())
#     lbl_pil.save(save_path)

img_root='/home/XJF/code/HRSID/data/images'
# coco_root = '/home/XJF/code/HRSID/data/masks'
# target_root='/home/XJF/code/HRSID/data/test/img'
target_root='/home/XJF/code/HRSID/data/all_imgs/img'
# mask_root = '/home/XJF/code/HRSID/data/test/mask'
annotation_file = '/home/XJF/code/HRSID/data/annotations/train2017.json'

save_iscrowd = False

coco = COCO(annotation_file)
catIds = coco.getCatIds()       # 类别ID列表
imgIds = coco.getImgIds()       # 图像ID列表
print("catIds len: {}, imgIds len: {}".format(len(catIds), len(imgIds)))

cats = coco.loadCats(catIds)   # 获取类别信息->dict
names = [cat['name'] for cat in cats]  # 类名称
print(names)

img_cnt = 0
crowd_cnt = 0

for idx, imgId in tqdm.tqdm(enumerate(imgIds), ncols=1000):
    if save_iscrowd:
        annIds = coco.getAnnIds(imgIds=imgId)      # 获取该图像上所有的注释id->list
    else:
        annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)  # 获取该图像的iscrowd==0的注释id
    
    if len(annIds) > 0:
        image = coco.loadImgs([imgId])[0]
        # print(image)
        # break
        ## ['coco_url', 'flickr_url', 'date_captured', 'license', 'width', 'height', 'file_name', 'id']
        fileName=image['file_name']
        h, w = image['height'], image['width']
        gt_name = image['file_name'].replace('.jpg', '.png')
        gt = np.zeros((h, w), dtype=np.uint8)
        anns = coco.loadAnns(annIds)    # 获取所有注释信息

        has_crowd_flag = 0
        save_flag = 0
        img=cv2.imread(os.path.join(img_root,fileName),-1)
        # print(img.shape)
        # break
        count=0
        for ann_idx, ann in enumerate(anns):
            cat = coco.loadCats([ann['category_id']])[0]
            cat = cat['name']
            cat = names.index(cat) + 1   # re-map

            if not ann['iscrowd']:  # iscrowd==0
                segs = ann['segmentation']
                # print(segs)
                # break
                boxs=ann['bbox']
                box=np.array(boxs).astype(np.int32)
                xmin= box[0]
                ymin= box[1]
                w= box[2]
                h= box[3]
                xmax=xmin+w
                ymax=ymin+h
                xmin_1=xmin
                xmax_1=xmax
                ymin_1=ymin
                ymax_1=ymax

                # xmin = max(0,xmin-15)
                # xmax = min(xmax+15,img.shape[1])
                # ymin = max(0,ymin-15)
                # ymax = min(ymax+15,img.shape[0])

                tmp_target=copy.deepcopy(img[ymin:ymax,xmin:xmax])
                # if ymin==0 and ymin_1-15<0:
                #     tmp_target=np.pad(tmp_target,((15-ymin_1,0),(0,0),(0,0)),mode='linear_ramp')
                
                # if ymax==img.shape[0] and ymax_1+15>img.shape[0]:
                #     # print('------')
                #     # print(img.shape[0]-int(obj["bndbox"]["ymax"]))
                #     # print(10-img.shape[0]+int(obj["bndbox"]["ymax"]))
                #     tmp_target=np.pad(tmp_target,((0,ymax_1+15-img.shape[0]),(0,0),(0,0)),mode='linear_ramp')
                
                # if xmin==0 and xmin_1-15<0:
                #     tmp_target=np.pad(tmp_target,((0,0),(15-xmin_1,0),(0,0)),mode='linear_ramp')
                
                # if xmax==img.shape[1] and xmax_1+15>=img.shape[1]:
                #     tmp_target=np.pad(tmp_target,((0,0),(0,xmax_1+15-img.shape[1]),(0,0)),mode='linear_ramp')
                
                target_path="%sTarget%02d.png"%(fileName[:-4],count)
                cv2.imwrite(os.path.join(target_root,target_path),tmp_target)
                # cv2.rectangle(gt,(xmin,ymin),(xmax,ymax),255,1)
                # for seg in segs:
                #     seg = np.array(seg).reshape(-1, 2)     # [n_points, 2]
                #     cv2.fillPoly(gt, seg.astype(np.int32)[np.newaxis, :, :], 255)
                #     tmp_gt=copy.deepcopy(gt[ymin_1:ymax_1,xmin_1:xmax_1])
                #     tmp_gt[tmp_gt<127]=0
                #     tmp_gt[tmp_gt>=127]=255
                #     tmp_gt=np.pad(tmp_gt,((15,15),(15,15)),mode='constant',constant_values=(0,0))
                    # cv2.imwrite(os.path.join(mask_root,target_path),tmp_gt)
                
                    # cv2.fillPoly(gt, box.astype(np.int32)[np.newaxis, :, :], 255)
                
            elif save_iscrowd:
                has_crowd_flag = 1
                rle = ann['segmentation']['counts']
                assert sum(rle) == ann['segmentation']['size'][0] * ann['segmentation']['size'][1]
                mask = coco.annToMask(ann)
                unique_label = list(np.unique(mask))
                assert len(unique_label) == 2 and 1 in unique_label and 0 in unique_label
                gt = gt * (1 - mask) + mask * 255   # 这部分填充255
            
            count+=1
        # save_path = os.path.join(coco_root, gt_name)
        # cv2.imwrite(save_path, gt)
        img_cnt += 1
        if has_crowd_flag:
            crowd_cnt += 1

        if idx % 1000 == 0:
            print('Processed {}/{} images.'.format(idx, len(imgIds)))

print('crowd/all = {}/{}'.format(crowd_cnt, img_cnt))
