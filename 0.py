# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 23:42:49 2018

@author: Administrator
"""

from skimage import data,filters,feature
from PIL import Image
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
import cv2
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])
img = array(Image.open("100.jpg").convert('L'))


#img = rgb2gray(img)
print(img.shape,img.dtype)
#edges = filters.sobel(img)
#plt.imshow(edges,plt.cm.gray)

edges = filters.sobel(img)
plt.imshow(edges,plt.cm.gray)
#edges1 = filters.roberts(img)
#plt.imshow(edges1)

filt_real, filt_imag = filters.gabor_filter(img,frequency=0.6) 
plt.figure('gabor',figsize=(8,8))

plt.subplot(121)
plt.title('filt_real')
plt.imshow(filt_real,plt.cm.gray)  

plt.subplot(122)
plt.title('filt-imag')
plt.imshow(filt_imag,plt.cm.gray)

#plt.show()


image = cv2.imread("100.jpg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)#将图像转化为灰度图像
cv2.imshow("Original",image)
cv2.waitKey()

#Sobel边缘检测
sobelX = cv2.Sobel(image,cv2.CV_64F,1,0)#x方向的梯度
sobelY = cv2.Sobel(image,cv2.CV_64F,0,1)#y方向的梯度

sobelX = np.uint8(np.absolute(sobelX))#x方向梯度的绝对值
sobelY = np.uint8(np.absolute(sobelY))#y方向梯度的绝对值

sobelCombined = cv2.bitwise_or(sobelX,sobelY)#
cv2.imshow("Sobel X", sobelX)
#cv2.waitKey()
cv2.imshow("Sobel Y", sobelY)
#cv2.waitKey()
cv2.imshow("Sobel Combined", sobelCombined)
#cv2.waitKey()