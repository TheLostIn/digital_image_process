# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 11:12:41 2018

@author: Administrator
"""
import cv2
import numpy as np
def create_mask(img, cnt):
    '''Create a mask of the same size as the image
       based on the interior of the contour.'''
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    print("create_mask, cnt=%s" % cnt)
    cv2.drawContours(mask, [cnt], 0, (0, 255, 0), -1)
    return mask

page_mask = create_mask(raw, page_contour)
print("Creating mask from contour %s, on raw shape %s" % (page_contour, raw.shape))
