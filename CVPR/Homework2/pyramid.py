# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# def gaussian(original_image,down_times=5):
#     temp = original_image.copy()
#     gaussian_pyramid = [temp]
#     for i in range(down_times):
#         temp = cv2.pyrDown(temp)
#         gaussian_pyramid.append(temp)
#     return gaussian_pyramid
# if __name__ == "__main__":
#     a = cv2.imread("1.jpeg", -1)
#     gaussian_pyramid = gaussian(a, down_times=5)
#     plt.subplot(2, 3, 1), plt.imshow(a, cmap='gray')
#     plt.subplot(2, 3, 2), plt.imshow(gaussian_pyramid[2], cmap='gray')
#     plt.subplot(2, 3, 3), plt.imshow(gaussian_pyramid[4], cmap='gray')
#     plt.show()
#     print(gaussian_pyramid[0].shape, gaussian_pyramid[1].shape, gaussian_pyramid[5].shape)


import numpy as np
import cv2
import matplotlib.pyplot as plt
def gaussian(original_image,down_times):
    temp = original_image.copy()
    gaussian_pyramid = [temp]
    for i in range(down_times):
        temp = cv2.pyrDown(temp)
        gaussian_pyramid.append(temp)
    return gaussian_pyramid
def laplacian(gaussian_pyramid, up_times):
    laplacian_pyramid = [gaussian_pyramid[-1]]
    for i in range(up_times, 0, -1):
        print(i)
        temp_pyrUp = cv2.pyrUp(gaussian_pyramid[i])
        temp_lap = cv2.subtract(gaussian_pyramid[i-1], temp_pyrUp)
        laplacian_pyramid.append(temp_lap)
    return laplacian_pyramid
if __name__ == "__main__":
    a = cv2.imread("2.jpeg", -1)
    gaussian_pyramid = gaussian(a, down_times=5)
    laplacian_pyramid = laplacian(gaussian_pyramid, up_times=5)
    plt.subplot(2, 3, 1), plt.imshow(a, cmap='gray')
    plt.subplot(2, 3, 2), plt.imshow(gaussian_pyramid[2], cmap='gray')
    plt.subplot(2, 3, 3), plt.imshow(gaussian_pyramid[4], cmap='gray')
    plt.subplot(2, 3, 4), plt.imshow(laplacian_pyramid[0], cmap='gray')
    plt.subplot(2, 3, 5), plt.imshow(laplacian_pyramid[1], cmap='gray')
    plt.subplot(2, 3, 6), plt.imshow(laplacian_pyramid[5], cmap='gray')
    plt.show()
    print(gaussian_pyramid[0].shape, len(gaussian_pyramid), len(laplacian_pyramid))
