import  numpy as np
import cv2

def  GaussKernel(sig=None,m=0):
    if m == 0:
        m = int(sig * 2 * 3 + 1)
        print('计算的m',m)
    w = np.zeros((m, m), dtype=np.float)
    middle_m = m//2
    #生成高斯核
    for x in range(-middle_m, - middle_m + m):
        for y in range(-middle_m, - middle_m + m):
            w[y + middle_m, x + middle_m] = np.exp(-(x ** 2 + y ** 2) / (2 * (sig ** 2)))
    w /= (sig * np.sqrt(2 * np.pi))
    #归一化
    w /= w.sum()
    return w

def normalize(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def filter(f, w, method):
    x, y = w.shape
    fh, fw = f.shape
    nh = fh + x - 1
    nw = fw + y - 1
    add_h = int(x) // 2
    add_w = int(y) // 2
    n = np.zeros((nh, nw))
    g = np.zeros((fh, fw))
    n[add_h:nh - add_h, add_w:nw - add_w] = f
    if method == 'replicate':
        n[0:add_h,add_w:nw-add_w] = f[0,:]
        n[nh-add_h:,add_w:nw-add_w] = f[-1,:]
        for i in range(add_w):
            n[:,i] = n[:,add_w]
            n[:,nw-1-i] = n[:,nw-1-add_w]
        for i in range(fh):
            for j in range(fw):
                g[i,j] = np.sum(n[i:i+x,j:j+y] * w)
        g = g.clip(0,1)
        return g
    if method == 'zero':
        for i in range(fh):
            for j in range(fw):
                g[i,j] = np.sum(n[i:i+x,j:j+y] * w)
        g = g.clip(0,1)
        return g

if __name__ == '__main__':
    sig = [1, 2, 3, 5]
    # 图片导入
    f1= cv2.imread("8.jpeg",cv2.IMREAD_GRAYSCALE)

    for i in sig:
        w = GaussKernel(i)
        g1 = filter(normalize(f1), w, method='replicate')
        cv2.imshow('sig='+str(i)+',1_re', g1)


    #对比像素复制和补零下滤波结果在边界上的差别
    for i in sig:
        w = GaussKernel(i)
        g1 = filter(normalize(f1), w, method='replicate')
        g3 = filter(normalize(f1), w,method='zero')
        cv2.imshow('sig='+str(i)+',1_0', g3)
        cv2.imshow('sig=' + str(i) + ',8_diff', np.abs(g3-g1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
