import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
image=cv2.imread('3.jpeg')
print(image.shape)
im0=image[:,:,0]
im1=image[:,:,1]
im2=image[:,:,2]


k=15
def Gaussmodel(k,delta):
    row=2*k+1
    col=2*k+1
    A=np.zeros((row,col))
    for i in range(row):
        for j in range(row):
            fenzi=(i+1-k-1)**2+(j+1-k-1)**2
            print(fenzi)
            A[i,j]=np.exp(-fenzi/(2*delta))/(2*np.pi*delta)
    print(A)
    A=A/A[0,0]
    print(A)
    A=A/A.sum()
    return A

A=Gaussmodel(int((k-1)/2),2)
print(A)
def GaussFilter(im0):
    row_im=im0.shape[0]
    col_im=im0.shape[1]
    tmp=np.zeros((row_im+k-1,col_im+k-1))
    tmp[int((k-1)/2):-int((k-1)/2),int((k-1)/2):-int((k-1)/2)]=im0

    tar_im=np.zeros((row_im,col_im))
    tar_imtmp=np.zeros((A.shape[0],A.shape[1]))
    for i in range(row_im):
        for j in range(col_im):
            for m1 in range(A.shape[0]):
                for m2 in range(A.shape[1]):
                    tar_imtmp[m1][m2]=tmp[int(i+m1)][int(j+m2)]*A[m1][m2]

            tar_im[i][j]=tar_imtmp.sum()
    tar_im=np.dot(tar_im,1/tar_im.max()*256)

    tar_im=np.array(tar_im,dtype='uint8')
    return(tar_im)

tar_im0=GaussFilter(im0)
tar_im1=GaussFilter(im1)
tar_im2=GaussFilter(im2)

cv2.imshow("00",image)
cv2.imshow("0",im0)
cv2.imshow("1",im1)
cv2.imshow("2",im2)

cv2.imshow("t0",tar_im0)
cv2.imshow("t1",tar_im1)
cv2.imshow("t2",tar_im2)

tar_im=cv2.merge([tar_im0,tar_im1,tar_im2])
cv2.imshow("t",tar_im)
cv2.waitKey(0)
