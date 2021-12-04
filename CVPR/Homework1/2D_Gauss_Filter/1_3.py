import numpy as np
def gauss_high_pass_filter(source, center, radius=5):
    """
    create gaussian high pass filter 
    param: source: input, source image
    param: center: input, the center of the filter, where is the lowest value, (0, 0) is top left corner, source.shape[:2] is 
                   center of the source image
    param: radius: input, the radius of the lowest value, greater value, bigger blocker out range, if the radius is 0, then all
                   value is 1
    return a [0, 1] value filter
    """
    M, N = source.shape[1], source.shape[0]
    
    u = np.arange(M)
    v = np.arange(N)
    
    u, v = np.meshgrid(u, v)
    
    D = np.sqrt((u - center[1]//2)**2 + (v - center[0]//2)**2)
    D0 = radius
    
    if D0 == 0:
        kernel = np.ones(source.shape[:2], dtype=np.float)
    else:
        kernel = 1 - np.exp(- (D**2)/(D0**2))   
    return kernel

# 使用高能滤波和阈值处理增强图像
import cv2
import numpy as np
import matplotlib.pyplot as plt
 
img_finger = cv2.imread('8_blur.jpeg', 0) #直接读为灰度图像

plt.figure(figsize=(15, 12))
plt.subplot(221),plt.imshow(img_finger,'gray'),plt.title('origial')

#--------------------------------
fft = np.fft.fft2(img_finger)
fft_shift = np.fft.fftshift(fft)
amp_img = np.abs(np.log(1 + np.abs(fft_shift)))
plt.subplot(222),plt.imshow(amp_img,'gray'),plt.title('FFT')

#--------------------------------
BHPF = gauss_high_pass_filter(img_finger, img_finger.shape, radius=30)
plt.subplot(223),plt.imshow(BHPF,'gray'),plt.title('BHPF')

#--------------------------------
f1shift = fft_shift * BHPF
f2shift = np.fft.ifftshift(f1shift) #对新的进行平移
img_back = np.fft.ifft2(f2shift)


# 出来的是复数，无法显示
img_new = np.abs(img_back)

# 调整大小范围便于显示
img_new = (img_new-np.amin(img_new))/(np.amax(img_new)-np.amin(img_new))
plt.subplot(224),plt.imshow(img_new,'gray'),plt.title('After BHPF')

plt.tight_layout()
plt.show()

# import  cv2
# def bi_demo(image):#高斯双边滤波
#     dst = cv2.bilateralFilter(src=image, d=0, sigmaColor=100, sigmaSpace=15)
#     cv2.namedWindow('bi_demo',0)
#     cv2.resizeWindow('bi_demo',600,800)
#     cv2.imshow("bi_demo", dst)

# src = cv2.imread('8.jpeg')
# bi_demo(src)
# cv2.namedWindow('src', 0)
# cv2.resizeWindow('src', 600, 800)
# cv2.imshow('src',src)
# plt.show()
# cv2.waitKey(0)

