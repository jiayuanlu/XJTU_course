import cv2
import numpy as np
 
#读入图像并转化为float类型，用于传递给harris函数
# filename = '1.jpeg'
 
img = cv2.imread('1.jpeg')
 
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
gray_img = np.float32(gray_img)
 
#对图像执行harris
'''
• img - 数据类型为 float32 的输入图像。
　　• blockSize - 角点检测中要考虑的领域大小。
　　• ksize - Sobel 求导中使用的窗口大小
　　• k - Harris 角点检测方程中的自由参数,取值参数为 [0,04,0.06].
'''
# Harris_detector = cv2.cornerHarris(gray_img, 2, 1, 0.04)
# Harris_detector = cv2.cornerHarris(gray_img, 2, 3, 0.04)
Harris_detector = cv2.cornerHarris(gray_img, 2, 31, 0.04)
# Harris_detector = cv2.cornerHarris(gray_img, 2, 7, 0.04)
# Harris_detector = cv2.cornerHarris(gray_img, 2, 9, 0.04)
 
#腐蚀harris结果
dst = cv2.dilate(Harris_detector, None)
 
# 设置阈值
threshold1 = 0.01*dst.max()
 
img[dst > threshold1] = [255,0,0]
 
cv2.imshow('show', img)
cv2.imwrite('Harris31.jpg',img)
cv2.waitKey()