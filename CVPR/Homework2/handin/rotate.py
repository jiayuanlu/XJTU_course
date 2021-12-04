import cv2
from matplotlib import pyplot as plt
from pathlib import WindowsPath
import numpy as np
from scipy import signal
from scipy import misc
from PIL import Image
import sys
import math
image=cv2.imread('1.jpeg',0)
# img=Image.open('1.jpeg')
# image=img.convert('L')
high,wide=image.shape
theta=-math.radians(30)
xuanzhuan=np.zeros((high,wide),dtype=np.uint8)


#旋转
trans1=np.array([[math.cos(theta),-math.sin(theta),0],
                        [math.sin(theta), math.cos(theta),0],
                        [    0,              0,         1]])

#平移
trans2=np.array([[1,0,high/4],[0,1,wide/4],[0,0,1]])
#缩放
trans3=np.array([[1.5,0,0],[0,1.5,0],[0,0,1]])
for x0 in range(high):
    for y0 in range(wide):
        pos=np.array([x0-high/4,y0-wide/4,1])
        # trans=np.dot(trans2,trans1)
        [x,y,z]=np.dot(trans1,pos)#旋转
        [x,y,z]=np.dot(trans2,[x,y,z])#旋转+平移=欧式
        [x,y,z]=np.dot(trans3,[x,y,z])#旋转+平移+缩放=相似
        # [x,y,z]=np.dot(trans1,pos)
        x=int(x)
        y=int(y)
        if x>=high or y>=wide or x<0 or y<0:
            xuanzhuan[x0][y0]=255
        else:
            xuanzhuan[x0][y0]=image[x][y]
cv2.imshow('rotate.jpg',xuanzhuan)
cv2.imwrite('xiangsi.jpg',xuanzhuan)
cv2.waitKey(0)
cv2.destroyAllWindows()
