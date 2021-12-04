# import  numpy as np
# import cv2
# from sympy import *
# import matplotlib.pyplot as plt

# image=plt.imread('1.jpeg')

# def normalize(data):
#     _range = np.max(data) - np.min(data)
#     return (data - np.min(data)) / _range

# # def  GaussKernel(sig=5,m=0):
# #     if m == 0:
# #         m = int(sig * 2 * 3 + 1)
# #         print('计算的m',m)
# #     w = np.zeros((m, m), dtype=np.float)
# #     middle_m = m//2
# #     #生成高斯核
# #     for x in range(-middle_m, - middle_m + m):
# #         for y in range(-middle_m, - middle_m + m):
# #             w[y + middle_m, x + middle_m] = np.exp(-(x ** 2 + y ** 2) / (2 * (sig ** 2)))
# #     w /= (sig * np.sqrt(2 * np.pi))
# #     #归一化
# #     w /= w.sum()
# #     f=diff(w,x,1)
# #     return f
# # w=GaussKernel()
# # f=diff(w,x,1)
# def gaussian_kernel_2d(ksize, sigma):
#     w=cv2.getGaussianKernel(ksize, sigma) * cv2.getGaussianKernel(ksize,sigma).T
#     f=diff(w,)
# 	return cv2.getGaussianKernel(ksize, sigma) * cv2.getGaussianKernel(ksize,sigma).T
# w=gaussian_kernel_2d(3,1)

# result=cv2.filter2D(image,-1,f)

# plt.imshow(result)
# cv2.imwrite('Gauss_grad.jpg',result)



from PIL import Image
from numpy import *
import matplotlib.pyplot as plt
from scipy.ndimage import filters
im = array(Image.open('1.jpeg').convert('L'))
# im = array(Image.open('1.jpeg'))
sigma = 7  # 标准差
imx = zeros(im.shape)
filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
imy = zeros(im.shape)
filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
plt.subplot(1, 3, 1)
plt.axis('off')
plt.imshow(im, plt.cm.gray)
plt.imsave('Gauss_g1_7.jpg',im)
plt.subplot(1, 3, 2)
plt.axis('off')
plt.imshow(imx, plt.cm.gray)
plt.imsave('Gauss_gx_7.jpg',imx)
plt.subplot(1, 3, 3)
plt.axis('off')
plt.imshow(imy, plt.cm.gray)
plt.imsave('Gauss_gy_7.jpg',imy)
plt.show()
