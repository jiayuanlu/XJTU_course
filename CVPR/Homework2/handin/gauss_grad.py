import matplotlib.pyplot as plt
import numpy as np
import math
import cv2
from numpy.lib.function_base import diff

sigma=7
# size=int(sigma * 2 * 3 + 1)
# kernel = np.zeros([size], dtype=np.int)
image=plt.imread('1.jpeg')
sum=0
gauss=np.zeros([5,5])
k=2#5-3=2
for i in range(5):
    for j in range(5):
        gauss[i,j] = math.exp(-1/2 * (np.square(i-3)/np.square(sigma)           #生成二维高斯分布矩阵
                        + (np.square(j-3)/np.square(sigma)))) / (2*math.pi*sigma*sigma)
        sum += gauss[i, j]
gauss /=sum
#         fenzi=(i+1-k-1)**2+(j+1-k-1)**2
#         gauss[i,j]=np.exp(-fenzi/(2*sigma))/(2*np.pi*sigma)
# gauss=gauss/gauss[0,0]
# gauss=gauss/gauss.sum()


gray=np.dot(image[...,:3], [0.299, 0.587, 0.114])
wide,high=gray.shape
gray1=np.zeros([wide-5,high-5])
for i in range(wide-5):
    for j in range(high-5):
        gray1[i,j]=np.sum(gray[i:i+5,j:j+5]*gauss)
wide1,high1=gray1.shape
dx=np.zeros([wide1-1,high1-1])
dy=np.zeros([wide1-1,high1-1])
grad=np.zeros([wide1-1,high1-1])
orientation=np.zeros([wide1-1,high1-1])
for i in range(wide1-1):
    for j in range(high1-1):
        dx[i,j]=gray1[i,j+1]-gray1[i,j]
        dy[i,j]=gray1[i+1,j]-gray1[i,j]
        grad[i,j]=np.sqrt(np.square(dx[i,j])+np.square(dy[i,j]))
        if dx[i,j]==0:
            orientation[i,j]=np.pi/2
        else:
            orientation[i,j]=math.atan(dy[i,j]/dx[i,j])
# cv2.imwrite('grad2.jpg',grad)
# cv2.imwrite("orientation2.jpg",orientation)

wide2,high2=grad.shape
NMS=np.copy(grad)
NMS[0,:]=NMS[wide2-1,:]=NMS[:,0]=NMS[:,high2-1]=0
for i in range(1,wide2-1):
    for j in range(1,high2-1):
        if grad[i,j]==0:
            NMS[i,j]=0
        else:
            grad_x=dx[i,j]
            grad_y=dy[i,j]
            gg=grad[i,j]
            G_x=np.abs(grad_x)
            G_y=np.abs(grad_y)
            if G_y>G_x:
                weight=G_x/G_y
                g1=grad[i-1,j]
                g2=grad[i+1,j]
                if grad_x*grad_y>0:
                    g3=grad[i-1,j-1]
                    g4=grad[i+1,j+1]
                else:
                    g3=grad[i-1,j-1]
                    g4=grad[i+1,j-1]
            else:
                weight=G_y/G_x
                g1=grad[i,j-1]
                g2=grad[i,j+1]
                if grad_x*grad_y>0:
                    g3=grad[i+1,j-1]
                    g4=grad[i-1,j+1]
                else:
                    g3=grad[i-1,j-1]
                    g4=grad[i+1,j+1]
            gg1=weight*g3+(1-weight)*g1
            gg2=weight*g4+(1-weight)*g2
            if gg>=gg1 and gg>=gg2:
                NMS[i,j]=gg
            else:
                NMS[i,j]=0
# cv2.imwrite("NMS1.jpg",NMS)

wide3,high3=NMS.shape
result=np.zeros([wide3,high3])
TL=0.2*np.max(NMS)
TH=0.3*np.max(NMS)
for i in range(1,wide3-1):
    for j in range(1,high3-1):
        if NMS[i,j]<TL:
            result[i,j]=0
        elif NMS[i,j]>TH:
            result[i,j]=1
        elif (NMS[i-1,j-1:j+1]<TH).any() or (NMS[i+1,j-1:j+1]).any() or (NMS[i,[j-1,j+1]]<TH).any():
            result[i,j]=1
# plt.imshow(result,camp='edge')
cv2.imwrite('edge.jpg',result)
# plt.show()
