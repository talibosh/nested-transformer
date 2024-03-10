import os
import cv2
import pandas as pd
import numpy as np

def get_seg_image(msk_path:str, img_path:str, tmp_sz:tuple[int,int] = (224,224), out_sz:tuple[int, int] = (56, 56)):
    if os.path.isfile(msk_path) == False:
        return np.zeros(1,1)
    mask_img = cv2.imread(msk_path)
    img = cv2.imread(img_path)
    #assert(mask_img.shape == img.shape)
    rszd_msk = cv2.resize(mask_img, out_sz, cv2.INTER_NEAREST)
    rszd_image = cv2.resize(img, out_sz, cv2.INTER_CUBIC)
    thresh = np.median(np.unique(rszd_msk))

    rszd_msk[rszd_msk < thresh] = 0
    rszd_msk[rszd_msk >= thresh] = 1

    seg_image = np.multiply(rszd_msk, rszd_image)
    x1, y1, w, h = cv2.boundingRect(rszd_msk[:,:,0])
    x2 = x1 + w
    y2 = y1 + h
    start = (x1, y1)
    end = (x2, y2)
    colour = (1, 0, 0)
    thickness = -1
    #self.bb_msk = self.np_msk.copy()
    #ff = cv2.rectangle(seg_image, start, end, colour, thickness)
    cut_seg = seg_image[y1:y2, x1:x2]
    #rszd_seg_np_image = cv2.resize(seg_image, out_sz, cv2.INTER_CUBIC)
    cut_seg = cv2.resize(cut_seg, out_sz, cv2.INTER_CUBIC)
    return cut_seg
