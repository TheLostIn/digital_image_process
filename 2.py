# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 10:52:18 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 18:00:22 2018

@author: fan
"""

import cv2
import numpy as np
'''
import os  
path='E:\project\digital_image_process\图像处理大作业\数据集\'
f=os.listdir(path)
n=0
for i in f:
    oldname=path+f[n]   
    newname=path+str(n)+'.jpg'
    os.rename(oldname,newname)
    n+=1
'''
def fan(i):
    # step1：加载图片，转成灰度图
    image = cv2.imread(i+".jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
    # step2:用Sobel算子计算x，y方向上的梯度，之后在x方向上减去y方向上的梯度，通过这个减法，我们留下具有高水平梯度和低垂直梯度的图像区域。
    gradX = cv2.Sobel(gray, cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1, ksize=-1)
     
    # subtract the y-gradient from the x-gradient
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)
   
    blurred = cv2.blur(gradient, (9, 9))
    _, thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)
    
    x = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    _a, cnts, _b = x
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
     
    rect = cv2.minAreaRect(c)
    box = np.int0(cv2.boxPoints(rect))
     
    cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
    result='a'+i+".jpg"
    cv2.imwrite(result, image)
    cv2.waitKey(0)
for i in range(20):
    fan(str(i))
