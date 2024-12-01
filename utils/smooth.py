import cv2
import skimage.measure as measure
import numpy as np

def smooth(pred):
    w,h=pred.shape
    if (w-30)*(h-30)<=1000:
        pred=cv2.medianBlur(pred, 3)
    elif (w-30)*(h-30)<=8000:
        if (w-30)/(h-30)<=1.2 and (w-30)/(h-30)>=0.8:
            pred=cv2.medianBlur(pred, 11)
        elif (w-30)/(h-30)>1.2:
            pred=cv2.medianBlur(pred, 7)
        else:
            pred=cv2.medianBlur(pred, 7)
        
    else:
        if (w-30)/(h-30)<=1.1 and (w-30)/(h-30)>=0.9:
            pred=cv2.medianBlur(pred, 15)
        elif (w-30)/(h-30)>1.1:
            pred=cv2.medianBlur(pred, 15)
        else:
            pred=cv2.medianBlur(pred, 15)
        
    labels = measure.label(pred,connectivity=2)

    properties = measure.regionprops(labels)
    valid_label = set()
    max_area = 0
    for prop in properties:
        if prop.area > max_area:
            max_area = prop.area
            max_label=prop.label
    current_bw = np.in1d(labels, np.array(max_label)).reshape(labels.shape)
    pred=current_bw.astype(np.uint8)
    if (w-30)*(h-30)<=1000:
        k=int((w-30)/(h-30)*7)+1
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, k))
        pred=cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
    elif (w-30)*(h-30)<=8000:
        k=int((w-30)/(h-30)*10)+1
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, k))
        pred=cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
    else:
        k=int((w-30)/(h-30)*21)+1
        kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, k))
        pred=cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)
    return pred