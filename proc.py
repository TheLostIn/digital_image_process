# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 14:59:29 2018

@author: Administrator
"""
 
from PIL import Image
import cv2  
import numpy as np
from matplotlib import pyplot as plt
import math
def pre_process(img):
    # 背景处理为白色
    # img = cv2.imread("10.jpg", 0) 
    #print(len(img))
    #print(img[0])
  #  cv2.imshow("new_im1g",img)
#    cv2.waitKey(0) 
    new_img=img
    for i in range(0,len(img)):
        for j in range(0,len(img[i])):
           # print(img[i][j])
            if(img[i][j]<127):
                new_img[i][j]=127
            elif(img[i][j]>210):
                new_img[i][j]=210
            else:
                new_img[i][j]=img[i][j]
    #cv2.imwrite("new_img2.jpg",new_img)
    #cv2.imshow("new_img",new_img)
   # cv2.waitKey(0) 
    return new_img

def find_canny(img):
    img = cv2.GaussianBlur(img, (3, 3), 10, 10)
    canny = cv2.Canny(img,30, 110)  

    #cv2.imshow('Canny', canny)
    #cv2.waitKey(0) 
   # cv2.imwrite("canny_test100_2.jpg",canny)
    return canny
def findmaxarea(img,count):
   #img = cv2.imread(name)
   
   #imgray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#   kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4, 1))
   

  # image, contours, hierarchy = cv2.findContours(erode_Image, cv2.RETR_TREE, \
 #                                                     cv2.CHAIN_APPROX_SIMPLE)
   ret,thresh = cv2.threshold(img,127,255,0)
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
   #     rect = cv2.minAreaRect(contours[i])
#        print(rect)
    #    cv2.rectangle(img,(x,y),(x+w,y+h),(200),2)
        if(w<h*3):
            cv2.rectangle(img,(x,y),(x+w,y+h),(127),2)
            max=i
            #print(x,y,w,h)
            '''       
             if(min_x>x):
                min_x=x
            if(min_y>y):
                min_y=y
            if(max_x<(x+w)):
                max_x=(x+w)
            if(max_y<(y+h)):
                max_y=(y+h)
                '''
           
      ##      cv2.imshow("contours", img)
       #     cv2.waitKey(0)
        #    if(max_area<w*h):
        #        max_area=w*h
       #         max=i
  # x,y,w,h = cv2.boundingRect(contours[max])
   #cv2.rectangle(img,(min_x,min_y),(max_x,max_y),(0,255,0),2)
  # cv2.imwrite("minarea1644.jpg",img)
   #print("w*h",w*h)
   #cv2.imshow("contours"+str(count), img)
   #cv2.waitKey(0)
   c=contours[max]
   x,y,w,h = cv2.boundingRect(c)
 #  cv2.destroyAllWindows()
   return {"x":x,"y":y,"w":w,"h":h,"w*h":w*h}
def showminarea(img,x,y,w,h,newfilename):
#    img = cv2.pyrDown(cv2.imread("100"+str(i)+".jpg", cv2.IMREAD_UNCHANGED))
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    #cv2.imwrite("minarea"+str(i)+".jpg",img)
    #cv2.imshow("contours", img)
    cv2.imwrite(newfilename,img)
   # cv2.waitKey(0)
    cv2.destroyAllWindows()
def dilate(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(4, 1))  
    dilated = cv2.dilate(img,kernel)
    #腐蚀图像  
    eroded = cv2.erode(dilated,kernel)  
#显示腐蚀后的图像  
  #  cv2.imshow("Eroded Image",eroded);  
  #  cv2.waitKey()
#    dilated = cv2.dilate(eroded,kernel)
   # dilated = cv2.dilate(dilated,kernel)
 #   eroded = cv2.erode(dilated,kernel) 
  #  eroded = cv2.erode(eroded,kernel) 
   # dilated = cv2.dilate(eroded,kernel)
 #   dilated = cv2.dilate(dilated,kernel)
    dilated = cv2.dilate(dilated,kernel)
    dilated = cv2.dilate(dilated,kernel)
  #  dilated = cv2.dilate(dilated,kernel)
  #  dilated = cv2.dilate(dilated,kernel)
    #显示膨胀后的图像  
    #cv2.imshow("Dilated Image",dilated) 
    #cv2.waitKey()
    return dilated
def find_serial_area(img,min_area):
       #img = cv2.pyrDown(cv2.imread("100"+str(count)+".jpg", cv2.IMREAD_UNCHANGED))
   # img = cv2.pyrDown(cv2.imread(name, cv2.IMREAD_UNCHANGED))
#   img = cv2.imread(name)
 #  img = dilate(img)
 #  img = cv2.cvtColor(imgray,cv2.COLOR_BGR2GRAY)
 #  ret,thresh = cv2.threshold(img,127,255,0)
 
   ret,thresh = cv2.threshold(img,1,255,cv2.THRESH_BINARY) 
  # print(imgray[0])

   image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  # print(thresh)
   img = thresh
   max = 0
   max_area=1
   min_x=2000
   min_y=2000
   max_x=0
   max_y=0
 #  print("find_serial_num")

   for i in range(0,len(contours)):
        c=contours[i]
        x,y,w,h = cv2.boundingRect(c)
      #  print(min_area["rect"][0][1],min_area["rect"][0][0])
      #  print("y",y,min_area["rect"][0][1]-min_area["h"]/2+50)
     #   print(x,y,w,h,min_area["rect"][0][1]-min_area["h"]/2+50)
       # cv2.rectangle(img,(x,y),(x+w,y+h),(127),2)
       # print(min_area["rect"][0][0])
      #  a = abs((x+w/2)-min_area["rect"][0][0])
     #   print(a)
        
        if(w>h*2 and w>(min_area["w"]*0.18) and w<(min_area["w"]*0.4) and h>15 and abs((x+w/2)-min_area["rect"][0][0])>100):
            
            cv2.rectangle(img,(x,y),(x+w,y+h),(127),2)
          #  print("yq",y,min_area["rect"][0][1]-min_area["h"]/2+40)
            if(y>(min_area["rect"][0][1]-min_area["h"]/2+40) and h<100 ):
               print("y123123q",y,min_area["rect"][0][1]-min_area["h"]/2+40,h)
             #  print(x,y,w,h)
               max=i
        '''
        if(w>h*4 and h>(min_area['h']/20)):
            print("in",i,x,y,w,h,min_area['y']+200,y)
            cv2.rectangle(img,(x,y),(x+w,y+h),(250),2)
            
            mid_x = int(x+w/2)
            mid_y = int(y+h/2)
            count_black = 0
       #     cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
            if(y>(min_area['y']+80)):
              cv2.rectangle(img,(x,y),(x+w,y+h),(200),2)
              max=i
              print('find')
            #for j in range(x,x+w):
          #      for k in range(y,y+h):
             #        if(img[j][k]==0):
           #              count_black = count_black + 1
                         #print(img[j][k])
               
           # if(count_black<(w*h)/3):
          #      cv2.rectangle(img,(x,y),(x+w,y+h),(200),2)
          #      max=i
           #     print('find')
             #   print(w,h)
            print(i)
            print(i,x,y,w,h)
      #      print(i,count_black,w*h)
            print(i,mid_x,mid_y)
            print(i,img[mid_x][mid_y])
           # if(img[mid_x][mid_y]==[  0  , 0 ,  0, 255]):
         '''   

  # print("max",max)
   x,y,w,h = cv2.boundingRect(contours[max])
   #cv2.rectangle(img,(min_x,min_y),(max_x,max_y),(0,255,0),2)
   #cv2.imwrite("minarea"+str(100)+".jpg",img)
  # cv2.imshow("contours12", img)
 #  cv2.waitKey()
   #cv2.waitKey(0)
  # cv2.destroyAllWindows()
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
#    img = img.convert('RGBA')
    img = img.rotate(angle,expand=1)
    p = Image.new('RGBA', img.size,(0,)*4)
    out = Image.composite(img, p, img)
    img = np.array(out)
   # cv2.imshow('img_rotate',img)
  #  cv2.waitKey()
    return img
def findminrotate1(img,start_num,num):
    # im1 = Image.open(canny_pic_src)
  #  cv2.imshow('canny',img)
  #  cv2.waitKey()
    min_area={}
    new_area={}
    
    canny_Image = cv2.Canny(img, 10, 140)
    #cv2.imshow("canny", canny_Image)
    dilate_Image = cv2.dilate(canny_Image, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60)))
    erode_Image = cv2.erode(dilate_Image, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60)))
    #cv2.imshow("erode", erode_Image)
    #cv2.waitKey(0)
    
    for i in range(start_num,num):

        img1 = rotateimg(erode_Image,i)
        #cv2.imshow('canny'+str(i),img)
        #cv2.waitKey()
        new_area = findmaxarea(erode_Image,i)
  #      print(i,new_area)
  #      print(min_area)
        if(i==start_num):
            min_area = new_area
            min_area['angle'] = i
        elif ((int(new_area['w'])>int(new_area['h'])) and (int(min_area['w'])*int(min_area['h'])>int(new_area['w'])*int(new_area['h']))):
            new_area['angle'] = i
            min_area = new_area
         #   print(min_area)

    print(min_area)
    return min_area
def findminrotate(img,start_num,num):
    #  cv2.waitKey()
    min_area={}
    new_area={}
    
    canny_Image = cv2.Canny(img, 10, 140)
  #  cv2.imshow("canny", canny_Image)
    dilate_Image = cv2.dilate(canny_Image, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60)))
    erode_Image = cv2.erode(dilate_Image, cv2.getStructuringElement(cv2.MORPH_RECT, (60, 60)))
 #   cv2.imshow("erode", erode_Image)
 #   cv2.waitKey(0)
    image, contours, hierarchy = cv2.findContours(erode_Image, cv2.RETR_TREE, \
                                                      cv2.CHAIN_APPROX_SIMPLE)
  #  print(contours)
    new_area = findmaxarea(erode_Image,122)
   # rect = cv2.minAreaRect(c)  # 计算最小矩形区域
    # calculate coordinates of the minimum area rectangle
 #   box = cv2.boxPoints(rect)
    # normalize coordinates to integers
  #  box = np.int0(box)
    # draw contours
  #  cv2.drawContours(img, [box], 0, (0,0, 255), 3) # 画出这个矩形
    info={}
    for i in range(len(contours)):
         c=contours[i]
         x,y,w,h = cv2.boundingRect(c)
         rect = cv2.minAreaRect(contours[i])
         box = cv2.boxPoints(rect)
         box = np.int0(box)
         #cv2.drawContours(img, [box], 0, (127), 3)
        # print(rect[1][0],rect[1][1])
         if(rect[1][0]>rect[1][1]):
                max_1 = rect[1][0]
                min_1 = rect[1][1]
         else:
                max_1 = rect[1][1]
                min_1 = rect[1][0]
         if(max_1<min_1*3 and min_1>188 and max_1>300):
           #  print(rect[1][0],rect[1][1])
           #  cv2.drawContours(img, [box], 0, (127), 3)
             
             info["w"] = max_1
             info["h"] = min_1
             info["box"] = box
             info["rect"] = rect
     #    print(info)
    rows, cols = erode_Image.shape  
  #  cv2.imshow("Image", img)
 #   cv2.waitKey(0)
    rect = info["rect"]
    if (rect[1][0] >= rect[1][1]):
        angle = rect[2]  # 负数，顺时针旋转
    else:
        angle = 90 - np.fabs(rect[2])  # 正数，逆时针旋转
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)  
    img = cv2.warpAffine(img, M, (cols, rows))  
    info["img"] = img
    
  
   # cv2.imshow("rotation", img)
   # cv2.waitKey()  
    cv2.destroyAllWindows()  
         #cv2.rectangle(img,(x,y),(x+w,y+h),(127),2)
    #     if(w<h*3):
    #         cv2.rectangle(img,(x,y),(x+w,y+h),(127),2)
    #         print(x,y,w,h)
     ##        cv2.imshow("Image", img)
    #         cv2.waitKey(0)
        #    if (rect[1][0] > rect[1][1]):
          #      maxer = rect[1][0]
          #      miner = rect[1][1]
          #  else:
          #      maxer = rect[1][1]
          #      miner = rect[1][0]
          #  if miner > 150 and maxer > 300:
          #      w=maxer
          #      h=miner
   # print(w,h)

    return info
                
def find_serial_num(img,min_area):
    img=dilate(img)
  #  img=dilate(img)
  #  img=dilate(img)
    serial_area = find_serial_area(img,min_area)
 #   newfilename = "ok/origin_"+filename
   # showminarea(img,serial_area['x'],serial_area['y'],serial_area['w'],serial_area['h'])
    return serial_area
def saveprocessed(img,min_area,info,newfilename):
#    img = cv2.pyrDown(cv2.imread("100"+str(i)+".jpg", cv2.IMREAD_UNCHANGED))
  #  cv2.rectangle(img,(min_area['x'],min_area['y']),(min_area['x']+min_area['w'],min_area['y']+min_area['h']),(0,255,0),2)
    rect = info["rect"]
    if (rect[1][0] >= rect[1][1]):
        angle = info["angle"]  # 负数，顺时针旋转
    else:
        angle = 90 - np.fabs(info["angle"])  # 正数，逆时针旋转
    

    box = cv2.boxPoints(info["rect"])
    box = np.int0(box)
    cv2.drawContours(img, [box], 0, (0,255,0), 3)
    print(img.shape)
    rows, cols,num = img.shape  
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)  
    img = cv2.warpAffine(img, M, (cols, rows))
    cv2.rectangle(img,(info["serial_area"]['x'],info["serial_area"]['y']),(info["serial_area"]['x']+info["serial_area"]['w'],info["serial_area"]['y']+info["serial_area"]['h']),(0,255,0),2)
    #cv2.imwrite("minarea"+str(i)+".jpg",img)
    #cv2.imshow("contours", img)
    cv2.imwrite(newfilename,img)
    #cv2.waitKey(0)
  #  cv2.destroyAllWindows()
def main():
    min_area = {}
    fname = ["1","5","10","20","50","100","q","q (1)","q (2)","q (3)","q (4)","q (5)","q (6)","q (7)","q (8)","q (9)","q (10)","q (11)","q (12)","q (13)"];
    for i in range(0,len(fname)):
    #for i in range(0,16):
        print("i",i,fname[i])
        filename = "data/"+fname[i]+".jpg"
    #filename = "100.jpg"
        Ori_Image = cv2.imread(filename,0)
     #   resize_image = cv2.resize(Ori_Image, (640, 512))
        Gaussian_Image = cv2.GaussianBlur(Ori_Image, (3, 3), 10, 10)
     #   noise_Image = cv2.GaussianBlur(Resize_Image, (5, 5), 0, 0)
   ##     cv2.imshow("noise", noise_Image)
     #   Gray_Image = cv2.cvtColor(noise_Image, cv2.COLOR_BGR2GRAY)
  #      img = cv2.imread(filename,0)
 #       Resize_Image = cv2.resize(img, (640, 512))
        img3 = cv2.imread(filename)
     #   img3 = cv2.resize(img3, (640, 512))
        img1 = pre_process(Gaussian_Image)

        img1 = find_canny(img1)
        #cv2.imshow("img1",img1)
        #cv2.waitKey(0)

        info = findminrotate(img1,0,360)
        info["angle"] = info["rect"][2]
        img1 = info["img"]
        info1 = findminrotate(img1,0,360)
        info["serial_area"] = find_serial_num(info1["img"],info1)
        newfilename = "ok/"+filename;
        saveprocessed(img3,min_area,info,newfilename)
        
'''
#      img2 = rotateimg(img,min_area['angle'])
        img1 = rotateimg(img1,min_area['angle'])
        img3 = rotateimg(img3,min_area['angle'])
        # img1 = dilate(img1)
        serial_area = find_serial_num(img1,min_area)

        newfilename = "ok/"+filename;
        saveprocessed(img3,min_area,serial_area,newfilename)
'''
main()
cv2.destroyAllWindows() 