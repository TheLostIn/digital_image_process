# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:27:25 2018

@author: Administrator
"""

#coding=utf-8  
import cv2  
import numpy as np  

def dilate(Img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))  
    dilated = cv2.dilate(Img,kernel)
    #显示膨胀后的图像  
    cv2.imshow("Dilated Image",dilated);  
    return dilated




def findminarea(count,name):
   #img = cv2.pyrDown(cv2.imread("100"+str(count)+".jpg", cv2.IMREAD_UNCHANGED))
   # img = cv2.pyrDown(cv2.imread(name, cv2.IMREAD_UNCHANGED))
   img = cv2.imread(name)
   img = dilate(img)
   imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
   ret,thresh = cv2.threshold(imgray,127,255,0)
   image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   max = 0
   max_area=1
   min_x=2000
   min_y=2000
   max_x=0
   max_y=0
   for i in range(0,len(contours)):
        c=contours[i]
        x,y,w,h = cv2.boundingRect(c)
        if(w>3*h and h>40 and h<50):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        if(w>90 and h>90):
            print(x,y,w,h)
            if(min_x>x):
                min_x=x
            if(min_y>y):
                min_y=y
            if(max_x<(x+w)):
                max_x=(x+w)
            if(max_y<(y+h)):
                max_y=(y+h)
    
            #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            #cv2.imshow("contours", img)
            # cv2.waitKey(0)
            if(max_area<w*h):
                max_area=w*h
                max=i

   x,y,w,h = cv2.boundingRect(contours[max])
   #cv2.rectangle(img,(min_x,min_y),(max_x,max_y),(0,255,0),2)
   cv2.imwrite("minarea"+str(100)+".jpg",img)
   cv2.imshow("contours", img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   print(count,w*h)
   return [count,min_x,min_y,max_x-min_x,max_y-min_y]

findminarea(1001,"dilated_3.jpg")

cv2.waitKey(0)  
cv2.destroyAllWindows()  