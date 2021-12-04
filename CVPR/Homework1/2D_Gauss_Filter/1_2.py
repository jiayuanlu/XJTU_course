import cv2 as cv
import numpy as np
from numpy.core.fromnumeric import size


def Gauss(inputImage, outputImage, sigma,edge):
    timeBegin = cv.getTickCount()
    # 产生二维高斯滤波器kernel，行列不分离
    # 高斯滤波器大小为：size x size，size为奇数
    size=int(sigma * 2 * 3 + 1)
    kernel = np.zeros([size, size], dtype=np.int)
    center = size//2  # 将滤波器分为size x size个小方格，中心点为center，坐标为(0, 0)
    normal = 1/(np.exp(-(2*center**2)/(2*(sigma**2))))  # 用于整数化
    sum = 0  # 模板参数总和
    for i in range(size):
        for j in range(size):
            x = i-center  # 方格的横坐标
            y = center-j  # 方格的纵坐标
            kernel[i, j] = int(np.exp(-(x**2+y**2)/(2*sigma**2)) * normal)
            sum += kernel[i, j]
    # 对图像inputImage增添
    border = center  # 需要添加的边界大小
    if edge=="replicate":
        transImage = cv.copyMakeBorder(inputImage, border, border, border, border,
                                   borderType=cv.BORDER_REPLICATE)  # 复制最边缘像素
    elif edge=="zero":
        transImage=cv.copyMakeBorder(inputImage, border, border, border, border,
                                   borderType=cv.BORDER_DEFAULT) 
    elif edge=="reflect":
        transImage=cv.copyMakeBorder(inputImage, border, border, border, border,
                                   borderType=cv.BORDER_REFLECT)
    elif edge=="reflect_101":
        transImage=cv.copyMakeBorder(inputImage, border, border, border, border,
                                   borderType=cv.BORDER_REFLECT_101)
    
    # 开始平滑操作
    row, col, channel = inputImage.shape
    for i in range(row):
        for j in range(col):
            for k in range(channel):
                tmp = np.sum(np.multiply(transImage[i:i+size, j:j+size, k], kernel)) // sum
                if tmp < 0:
                    tmp = 0
                elif tmp > 255:
                    tmp = 255
                outputImage[i, j, k] = tmp
    timeEnd = cv.getTickCount()
    time = (timeEnd-timeBegin)/cv.getTickFrequency()
    return time


def SepGauss(inputImage, outputImage, sigma,edge):
    timeBegin = cv.getTickCount()
    # 产生一维高斯滤波器kernel，行列分离
    # 高斯滤波器大小为：size x size，size为奇数
    size=int(sigma * 2 * 3 + 1)
    kernel = np.zeros([size], dtype=np.int)
    center = size//2  # 将滤波器分为size x size个小方格，中心点为center，坐标为(0, 0)
    normal = 1/(np.exp(-center**2/(2*(sigma**2))))  # 用于整数化
    sum = 0  # 模板参数总和
    for i in range(size):
        kernel[i] = int(np.exp(-(i-center)**2/(2*sigma**2)) * normal)
        sum += kernel[i]
    kernelRow = kernel
    kernelCol = np.resize(kernel, (size, 1))
    print(kernelCol)
    # 对图像inputImage增添
    border = center  # 需要添加的边界大小
    if edge=="replicate":
        transImage = cv.copyMakeBorder(inputImage, border, border, border, border,
                                   borderType=cv.BORDER_REPLICATE)  # 复制最边缘像素
    elif edge=="zero":
        transImage=cv.copyMakeBorder(inputImage, border, border, border, border,
                                   borderType=cv.BORDER_DEFAULT) 
    elif edge=="reflect":
        transImage=cv.copyMakeBorder(inputImage, border, border, border, border,
                                   borderType=cv.BORDER_REFLECT)
    elif edge=="reflect_101":
        transImage=cv.copyMakeBorder(inputImage, border, border, border, border,
                                   borderType=cv.BORDER_REFLECT_101)
    
    
    # 开始平滑操作
    row, col, channel = inputImage.shape
    # 对行操作
    for j in range(col):
        for k in range(channel):
            tmp = np.sum(np.multiply(transImage[:, j:j+size, k], kernelRow), axis=1) // sum
            transImage[:, j+border, k] = tmp
    # 对列操作
    for i in range(row):
        for k in range(channel):
            tmp = np.sum(np.multiply(transImage[i:i + size, border:col + border, k], kernelCol), axis=0) // sum
            outputImage[i, :, k] = tmp

    timeEnd = cv.getTickCount()
    time = (timeEnd - timeBegin) / cv.getTickFrequency()
    return time


sig = float(input('Please input  sigma: '))
print('Please choose Gauss: ')
print('     1  Gauss')
print('     2  SeperateGauss')
print('     3  both')
flag = int(input('What you want is: '))

imgSrc = cv.imread('8.jpeg')  # (481, 641, 3)
imgout = np.zeros(list(imgSrc.shape), dtype='uint8')


time1 = 0  # 行列不分离的时间
time2 = 0  # 行列分离的时间
time = 0  # 两种方式的时间差

if flag == 1:
    time = Gauss(imgSrc, imgout, sig, edge="replicate")
elif flag == 2:
    time = SepGauss(imgSrc, imgout, sig, edge="replicate")
elif flag == 3:
    time1 = Gauss(imgSrc, imgout, sig, edge="replicate")
    time2 = SepGauss(imgSrc, imgout, sig, edge="replicate")

strSigma = 'Gaussian image(sigma: ' + str(sig) + ')'
cv.imshow('source image', imgSrc)
cv.imshow(strSigma, imgout)
saveSigma = str(sig) + '.png'
cv.imwrite(saveSigma, imgout)

if flag == 1 or flag == 2:
    print('time(s):', time)
elif flag == 3:
    print('time1(s):', time1)
    print('time2(s):', time2)
    print('time2-time1 =', time2-time1)

cv.waitKey(0)
cv.destroyAllWindows()
