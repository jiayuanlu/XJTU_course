import cv2
from matplotlib import pyplot as plt
import numpy as np
import sys
import math
from PIL import Image
image=cv2.imread('1.jpeg',1)
high,wide, channel=image.shape

pingyi=np.array([[1,0,wide/4],[0,1,high/4]],np.float32)
pingyi_trans=cv2.warpAffine(image,pingyi,(wide,high),borderValue=125)

xuanzhuan=cv2.getRotationMatrix2D((wide/2,high/2),30,1)
xuanzhuan_trans=cv2.warpAffine(image,xuanzhuan,(wide,high),borderValue=125)

x=0.2
y=0.3

def forward(image,x,y):
    row=image.shape[0]
    col=image.shape[1]
    img_forward=np.zeros((int(row+col*x),int(col+row*y),3),dtype=np.ubyte)
    for rows in range(row):
        for cols in range(col):
            img_forward[int(rows+cols*x),int(cols+rows*y),:]=image[rows,cols,:]
    return img_forward

def backward(image,x,y):
    row=image.shape[0]
    col=image.shape[1]
    img_backward=np.zeros(shape=image.shape,dtype=np.ubyte)
    for rows in range(row):
        for cols in range(col):
            bcol=int(cols-rows*y)
            brow=int(rows-cols*x)
            img_backward[brow,bcol,:]=image[rows,cols,:]
    return img_backward

fimg=forward(image,x,y)
bimg=backward(fimg,x,y)

down=cv2.pyrDown(image)
down1=cv2.pyrDown(down)
down2=cv2.pyrDown(down1)
down3=cv2.pyrDown(down2)

up=cv2.pyrUp(down3)
up1=cv2.pyrUp(up)
up2=cv2.pyrUp(up1)
up3=cv2.pyrUp(up2)

resizeh=high*2
resizew=wide*2

inter_Nearest=cv2.resize(down3,(resizew,resizeh),interpolation=cv2.INTER_NEAREST)
inter__doubleLinear=cv2.resize(down3,(resizew,resizeh),interpolation=cv2.INTER_LINEAR)

#相似变化为“欧式变换+均匀缩放”
#Similarity Transform（相似变换）(保角性) = Rotation（旋转） + Translation（平移） + Scale（放缩）
# img=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
similar=cv2.getRotationMatrix2D(center=(image.shape[0]/2,image.shape[1]/2),angle=30,scale=0.5)
img_similar=cv2.warpAffine(image,similar,(image.shape[0],image.shape[1]))

#不再具有保角性，具有保平行性
pt1=np.float32([[0,0],[wide,0],[0,high]])
pt2=np.float32([[wide*0.3,high*0.3],[wide*0.8,high*0.2],[wide*0.1,high*0.9]])
fangshe=cv2.getAffineTransform(pt1,pt2)
fangshe_trans=cv2.warpAffine(image,fangshe,(wide,high))

#不保角，不保平行，保直线性
pt3=np.float32([[0,0],[wide,0],[0,high],[wide,high]])
pt4=np.float32([[wide*0.1,high*0.1],[wide*0.9,wide*0.1],[high*0.2,high*0.8],[wide*0.7,high*0.7]])
touying=cv2.getPerspectiveTransform(pt3,pt4)
touying_trans=cv2.warpPerspective(image,touying,(wide,high))

#欧式变换为平移+旋转，图形大小以及形状不变
oushi_trans=cv2.warpAffine(pingyi_trans,xuanzhuan,(wide,high),borderValue=125)

def Gauss_pyramid(image,times=5):
    copy=image.copy()
    cv2.imshow('Gauss0',copy)
    cv2.imwrite('Gauss0.jpg',copy)
    img=[copy]
    for i in range(times):
        copy=cv2.pyrDown(copy)
        img.append(copy)
        cv2.imshow('Gauss'+str(i),copy)
        cv2.imwrite('Gauss'+str(i+1)+'.jpg',copy)
    return img

def Laplace_pyramid(image,times=5):
    img=[image[-1]]
    for i in range(times,0,-1):
        up0=cv2.pyrUp(image[i])
        laplace=cv2.subtract(image[i-1],up0)
        img.append(laplace)
    return img
        

# image1=cv2.imread('2.jpeg',-1)
# gauss=Gauss_pyramid(image1)
# laplace=Laplace_pyramid(gauss)
# plt.subplot(2, 3, 1),plt.imshow(laplace[0],cmap='gray')
# cv2.imwrite('Laplace0.jpg',laplace[0])
# plt.subplot(2, 3, 2), plt.imshow(laplace[1],cmap='gray')
# cv2.imwrite('Laplace1.jpg',laplace[1])
# plt.subplot(2, 3, 3),plt.imshow(laplace[2],cmap='gray')
# cv2.imwrite('Laplace2.jpg',laplace[2])
# plt.subplot(2, 3, 4),plt.imshow(laplace[3],cmap='gray')
# cv2.imwrite('Laplace3.jpg',laplace[3])
# plt.subplot(2, 3, 5),plt.imshow(laplace[4],cmap='gray')
# cv2.imwrite('Laplace4.jpg',laplace[4])
# plt.subplot(2, 3, 6),plt.imshow(laplace[5],cmap='gray')
# cv2.imwrite('Laplace5.jpg',laplace[5])
# plt.show()

# cv2.imshow('image',pingyi_trans)
# cv2.imwrite('pingyi.jpg',pingyi_trans)
# cv2.imshow('image1',xuanzhuan_trans)
# cv2.imwrite('xuanzhuan.jpg',xuanzhuan_trans)
# cv2.imshow('fimg',fimg)
# cv2.imwrite('forward.jpg',fimg)
# cv2.imshow('bimg',bimg)
# cv2.imwrite('backward.jpg',bimg)
# cv2.imshow('down',down)
# cv2.imwrite('down.jpg',down)
# cv2.imshow('down1',down1)
# cv2.imwrite('down1.jpg',down1)
# cv2.imshow('down2',down2)
# cv2.imwrite('down2.jpg',down2)
# cv2.imshow('down3',down3)
# cv2.imwrite('down3.jpg',down3)
# cv2.imshow('up',up)
# cv2.imwrite('up.jpg',up)
# cv2.imshow('up1',up1)
# cv2.imwrite('up1.jpg',up1)
# cv2.imshow('up2',up2)
# cv2.imwrite('up2.jpg',up2)
# cv2.imshow('up3',up)
# cv2.imwrite('up3.jpg',up3)
# cv2.imshow('inter_Nearest',inter_Nearest)
# cv2.imwrite('inter_Nearest.jpg',inter_Nearest)
# cv2.imshow('inter_DoubleLinear',inter__doubleLinear)
# cv2.imwrite('inter_DoubleLinear.jpg',inter__doubleLinear)
cv2.imshow('similar',img_similar)
cv2.imwrite('similar.jpg',img_similar)
cv2.imshow('fangshe',fangshe_trans)
cv2.imwrite('fangshe.jpg',fangshe_trans)
cv2.imshow('touying',touying_trans)
cv2.imwrite('touying.jpg',touying_trans)
# cv2.imshow('oushi',oushi_trans)
# cv2.imwrite('oushi.jpg',oushi_trans)

cv2.waitKey(0)
cv2.destroyAllWindows()


