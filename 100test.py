# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 20:00:17 2018

@author: Administrator
"""

#coding=utf-8  
from PIL import Image
import cv2  
import numpy as np

picsrc="E:\\project\\digital_image_process\\图像处理大作业\\数据集\\100.jpg"
def rotateimg(num):
    i=0
    for i in range(0,num):
        im1 = Image.open(picsrc)
        im2 = im1.rotate(i,expand=True)
        #im2.show()
        str1 = '100' + str(i)
        str1 = str1+".jpg"
        im2.save(str1)
def findminarea(count,name):
   #img = cv2.pyrDown(cv2.imread("100"+str(count)+".jpg", cv2.IMREAD_UNCHANGED))
   # img = cv2.pyrDown(cv2.imread(name, cv2.IMREAD_UNCHANGED))
   img = cv2.imread(name)
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
   cv2.rectangle(img,(min_x,min_y),(max_x,max_y),(0,255,0),2)
   cv2.imwrite("minarea"+str(100)+".jpg",img)
   cv2.imshow("contours", img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
   print(count,w*h)
   return [count,min_x,min_y,max_x-min_x,max_y-min_y]
def showminarea(i,x,y,w,h):
    img = cv2.pyrDown(cv2.imread("100"+str(i)+".jpg", cv2.IMREAD_UNCHANGED))
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imwrite("minarea"+str(i)+".jpg",img)
    cv2.imshow("contours", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    第三个参数method为轮廓的近似办法
        cv2.CHAIN_APPROX_NONE存储所有的轮廓点，相邻的两个点的像素位置差不超过1，即max（abs（x1-x2），abs（y2-y1））==1
        cv2.CHAIN_APPROX_SIMPLE压缩水平方向，垂直方向，对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需4个点来保存轮廓信息
        cv2.CHAIN_APPROX_TC89_L1，CV_CHAIN_APPROX_TC89_KCOS使用teh-Chinl chain 近似算法
    '''
'''
#100元的 ok

num=361
rotateimg(num)
L=[]
for i in range(0,num):
    L.append(findminarea(i))
min=0
min_area=999


for j in range(0,len(L)):
   # [i,x,y,w,h]=L[j]
    i=L[j][0]
    x=L[j][1]
    y=L[j][2]
    w=L[j][3]
    h=L[j][4]
    #print(i)
    if(j==1):
        min_area=w*h
        min=1
    else:
        if((w*h)<min_area and w>200 and h>200 and w>h):
            min_area = w*h
            min = j
    print(j,i,x,y,w,h,w*h,"minarea",min_area)
    #print(min_area)
print(min,L[min][1],L[min][2],L[min][3],L[min][4])
showminarea(L[min][0],L[min][1],L[min][2],L[min][3],L[min][4])
min=359
showminarea(L[min][0],L[min][1],L[min][2],L[min][3],L[min][4])
'''
img = cv2.imread("10.jpg", 0) 
print(len(img))
print(img[0])
new_img=img
for i in range(0,len(img)):
    for j in range(0,len(img[i])):
        if(img[i][j]<100):
            new_img[i][j]=225
        else:
            new_img[i][j]=img[i][j]
cv2.imwrite("new_img1.jpg",new_img)

img=new_img
img = cv2.GaussianBlur(img,(3,3),0)  
canny = cv2.Canny(img,10, 150)  

cv2.imshow('Canny', canny)
cv2.waitKey(0) 
cv2.imwrite("canny_test100_1.jpg",canny)
findminarea(174,"canny_test100_1.jpg")
cv2.waitKey(0)  
cv2.destroyAllWindows()  


