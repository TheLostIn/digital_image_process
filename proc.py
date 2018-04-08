# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:59:29 2018

@author: Administrator
"""
 
from PIL import Image
import cv2  
import numpy as np
from matplotlib import pyplot as plt

def pre_process(img):
    # 背景处理为白色
    # img = cv2.imread("10.jpg", 0) 
    #print(len(img))
    #print(img[0])
    #cv2.imshow("new_im1g",img)
    new_img=img
    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
            if(img[i][j]<100):
                new_img[i][j]=100
            else:
                new_img[i][j]=img[i][j]
    #cv2.imwrite("new_img2.jpg",new_img)
    #cv2.imshow("new_img",new_img)
    #cv2.waitKey(0) 
    return new_img

def find_canny(img):
    img = cv2.GaussianBlur(img,(3,3),0)  
    canny = cv2.Canny(img,20, 150)  

   # cv2.imshow('Canny', canny)
  #  cv2.waitKey(0) 
   # cv2.imwrite("canny_test100_2.jpg",canny)
    return canny
def findmaxarea(img,count):
   #img = cv2.imread(name)
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
        min_w=5
        min_h=5
        
        c=contours[i]
        x,y,w,h = cv2.boundingRect(c)
        
      #  cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        if(w>min_w and h>min_h):
          #  cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            #print(x,y,w,h)
            if(min_x>x):
                min_x=x
            if(min_y>y):
                min_y=y
            if(max_x<(x+w)):
                max_x=(x+w)
            if(max_y<(y+h)):
                max_y=(y+h)
    
           
            #cv2.imshow("contours", img)
            # cv2.waitKey(0)
            if(max_area<w*h):
                max_area=w*h
                max=i
  # x,y,w,h = cv2.boundingRect(contours[max])
   #cv2.rectangle(img,(min_x,min_y),(max_x,max_y),(0,255,0),2)
  # cv2.imwrite("minarea1644.jpg",img)
   #print("w*h",w*h)
  # cv2.imshow("contours"+str(count), img)
   #cv2.waitKey(0)
   cv2.destroyAllWindows()
   return {"x":min_x,"y":min_y,"w":max_x-min_x,"h":max_y-min_y,"w*h":(max_x-min_x)*(max_y-min_y)}
def showminarea(img,x,y,w,h):
#    img = cv2.pyrDown(cv2.imread("100"+str(i)+".jpg", cv2.IMREAD_UNCHANGED))
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #cv2.imwrite("minarea"+str(i)+".jpg",img)
    cv2.imshow("contours", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def dilate(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 1))  
    dilated = cv2.dilate(img,kernel)
    #腐蚀图像  
    eroded = cv2.erode(dilated,kernel)  
#显示腐蚀后的图像  
  #  cv2.imshow("Eroded Image",eroded);  
    cv2.waitKey()
    dilated = cv2.dilate(eroded,kernel)
   # dilated = cv2.dilate(dilated,kernel)
 #   eroded = cv2.erode(dilated,kernel) 
  #  eroded = cv2.erode(eroded,kernel) 
    dilated = cv2.dilate(eroded,kernel)
    dilated = cv2.dilate(dilated,kernel)
    dilated = cv2.dilate(dilated,kernel)
    dilated = cv2.dilate(dilated,kernel)
    dilated = cv2.dilate(dilated,kernel)
    dilated = cv2.dilate(dilated,kernel)
    #显示膨胀后的图像  
    cv2.imshow("Dilated Image",dilated) 
    cv2.waitKey()
    return dilated
def find_serial_area(img):
       #img = cv2.pyrDown(cv2.imread("100"+str(count)+".jpg", cv2.IMREAD_UNCHANGED))
   # img = cv2.pyrDown(cv2.imread(name, cv2.IMREAD_UNCHANGED))
#   img = cv2.imread(name)
 #  img = dilate(img)
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
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        if(w>h*4 and h>20):
            mid_x = int(x+w/2)
            mid_y = int(y+h/2)
            print(mid_x,mid_y)
            print(img[mid_x][mid_y])
           # if(img[mid_x][mid_y]==[  0  , 0 ,  0, 255]):
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            max=i
            print('find')
            print(w,h)

   x,y,w,h = cv2.boundingRect(contours[max])
   #cv2.rectangle(img,(min_x,min_y),(max_x,max_y),(0,255,0),2)
   #cv2.imwrite("minarea"+str(100)+".jpg",img)
   cv2.imshow("contours", img)
   #cv2.waitKey(0)
   cv2.destroyAllWindows()
   return {"x":x,"y":y,"w":w,"h":h}
def rotateimg(img,angle):
  #  rows,cols,channel = img.shape
    #在opencv中提供了cv2.getRotationMatrix2D函数获得变换矩阵。第一参数指定旋转圆点；第二个参数指定旋转角度；第三个参数指定缩放比例
    # M = cv2.getRotationMatrix2D((cols/2,rows/3),angle,1)
    # dst = cv2.warpAffine(img,M,(cols,rows))
    # cv2.imshow('img',dst)
    # cv2.waitKey(0)
    #im1 = Image.open(picsrc)
    #im2 = img.rotate(angle,expand=True)
    #im2.save(str1)

    img = Image.fromarray(img)
    img = img.convert('RGBA')
    img = img.rotate(angle,expand=1)
    p = Image.new('RGBA', img.size,(0,)*4)
    out = Image.composite(img, p, img)
    img = np.array(out)
   # cv2.imshow('img_rotate',img)
  #  cv2.waitKey()
    return img
def findminrotate(img,start_num,num):
    # im1 = Image.open(canny_pic_src)
    cv2.imshow('canny',img)
    cv2.waitKey()
    min_area={}
    new_area={}
    for i in range(start_num,num):

        img1 = rotateimg(img,i)
        #cv2.imshow('canny'+str(i),img)
        #cv2.waitKey()
        new_area = findmaxarea(img1,i)
        print(i,new_area)
        print(min_area)
        if(i==start_num):
            min_area = new_area
            min_area['angle'] = i
        elif ((int(new_area['w'])>int(new_area['h'])) and (int(min_area['w'])*int(min_area['h'])>int(new_area['w'])*int(new_area['h']))):
            new_area['angle'] = i
            min_area = new_area
         #   print(min_area)

    print(min_area)
    return min_area
def find_serial_num(img):
    img=dilate(img)
  #  img=dilate(img)
  #  img=dilate(img)
    serial_area = find_serial_area(img)
    showminarea(img,serial_area['x'],serial_area['y'],serial_area['w'],serial_area['h'])
def main():
    min_area = {}
    img = cv2.imread("20.jpg", 0)

    img1 = pre_process(img)
   
    img1 = find_canny(img1)

    min_area = findminrotate(img1,145,147)

 #   min_area = {'x': 409, 'y': 970, 'w': 253, 'h': 210, 'w*h': 53130, 'angle': 44}
    img2 = rotateimg(img,min_area['angle'])
    img1 = rotateimg(img1,min_area['angle'])
   # img1 = dilate(img1)
    serial_area = find_serial_num(img1)
    '''
  #  new_area = findmaxarea(img1)
    showminarea(img2,min_area['x'],min_area['y'],min_area['w'],min_area['h'])
    #showminarea(img1,serial_area['x'],serial_area['y'],serial_area['w'],serial_area['h'])
   # cv2.waitKey()
   # img1 = cv2.imread("canny_test100_1.jpg", 1)
   # minarea = findminrotate(img)
  #  img1 = rotateimg(img,minarea['angle'])
   # new_area = findmaxarea(img1)
   # showminarea(img1,minarea['x'],minarea['y'],minarea['w'],minarea['h'])
    
   # img1 = rotateimg(img,146)
  #  find_serial_num(img1)
   # new_area = findmaxarea(img1)
    # showminarea(img1,minarea['x'],minarea['y'],minarea['w'],minarea['h'])
  #  img=rotateimg(img,90)
   # cv2.imwrite("testsave.jpg",img)
'''
main()
cv2.destroyAllWindows() 