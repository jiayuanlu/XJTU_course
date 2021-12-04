# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
 
# img = cv2.imread('8.jpeg',0)
 
 
# dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
 
# magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
 
# plt.subplot(121),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
# plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
# plt.show()



import cv2
import numpy as np
import matplotlib.pyplot as plt

def fftImage(gray_img, row, col):
    rPadded = cv2.getOptimalDFTSize(row)
    cPadded = cv2.getOptimalDFTSize(col)
    imgPadded = np.zeros((rPadded, cPadded), np.float32)
    imgPadded[:row, :col] = gray_img
    fft_img = cv2.dft(imgPadded, flags=cv2.DFT_COMPLEX_OUTPUT)  #输出为复数，双通道
    return fft_img

def amplitudeSpectrum(fft_img):
    real = np.power(fft_img[:, :, 0], 2.0)
    imaginary = np.power(fft_img[:, :, 1], 2.0)
    amplitude = np.sqrt(real+imaginary)
    return amplitude

def graySpectrum(amplitude):
    amplitude = np.log(amplitude+1)
    spectrum = cv2.normalize(amplitude,  0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    spectrum *= 255
    return spectrum

def phaseSpectrum(fft_img):
    phase = np.arctan2(fft_img[:,:,1], fft_img[:, :, 0])
    spectrum = phase*180/np.pi  
    return spectrum

# 图像矩阵乘（-1）^(r+c), 中心化
def stdFftImage(img_gray, row, col):
    fimg = np.copy(img_gray)
    fimg = fimg.astype(np.float32)
    for r in range(row):
        for c in range(col):
            if(r+c)%2:
                fimg[r][c] = -1*img_gray[r][c]
    fft_img = fftImage(fimg, row, col)
    amplitude = amplitudeSpectrum(fft_img)
    ampSpectrum = graySpectrum(amplitude)
    return ampSpectrum

def GaussianHighFilter(image,d):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    def make_transform_matrix(d):
        transmatrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transmatrix.shape[0]):
            for j in range(transmatrix.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance(center_point,(i,j))
                transmatrix[i,j] = 1-np.exp(-(dis**2)/(2*(d**2)))
        return transmatrix
    d_matrix = make_transform_matrix(d)
    out_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
    return out_img

if __name__ == "__main__":
    img_gray = cv2.imread("8.jpeg", 0)
    row, col = img_gray.shape[:2]
    fft_img = fftImage(img_gray, row, col)
    amplitude = amplitudeSpectrum(fft_img)
    ampSpectrum = graySpectrum(amplitude)   
    phaSpectrum = phaseSpectrum(fft_img)   

    ampSpectrum_center = stdFftImage(img_gray, row, col) 
    cv2.imshow("img_gray", img_gray)
    cv2.imshow("ampSpectrum", ampSpectrum)
    cv2.imshow("ampSpectrum_center", ampSpectrum_center)
    cv2.imshow("phaSpectrum", phaSpectrum)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    s1 = np.log(np.abs(fft_img))
    img_d1 = GaussianHighFilter(img_gray,10)
    img_d2 = GaussianHighFilter(img_gray,30)
    img_d3 = GaussianHighFilter(img_gray,50)
    plt.subplot(131)
    plt.axis("off")
    plt.imshow(img_d1,cmap="gray")
    plt.title('D_10')
    plt.subplot(132)
    plt.axis("off")
    plt.title('D_30')
    plt.imshow(img_d2,cmap="gray")
    plt.subplot(133)
    plt.axis("off")
    plt.title("D_50")
    plt.imshow(img_d3,cmap="gray")
    plt.show()
