from sklearn import datasets
from sklearn.cluster import DBSCAN
import numpy as np
import random
import os
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import PIL.Image as Image
from matplotlib import animation
import operator
from functools import reduce

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


def findNeighbor(j,X,eps):
    N=[]
    for p in range(X.shape[0]):   #找到所有领域内对象
        temp=np.sqrt(np.sum(np.square(X[j]-X[p])))   #欧氏距离
        if(temp<=eps):
            N.append(p)
    return N


def dbscan(X,eps,min_Pts):
    k=-1
    NeighborPts=[]      #array,某点领域内的对象
    Ner_NeighborPts=[]
    fil=[]                                      #初始时已访问对象列表为空
    gama=[x for x in range(len(X))]            #初始时将所有点标记为未访问
    cluster=[-1 for y in range(len(X))]
    while len(gama)>0:
        j=random.choice(gama)
        gama.remove(j)  #未访问列表中移除
        fil.append(j)   #添加入访问列表

        NeighborPts=findNeighbor(j,X,eps)
        if len(NeighborPts) < min_Pts:
            cluster[j]=-1   #标记为噪声点
        else:
            k=k+1
            cluster[j]=k
            for i in NeighborPts:
                if i not in fil:
                    gama.remove(i)
                    fil.append(i)
                    Ner_NeighborPts=findNeighbor(i,X,eps)
                    if len(Ner_NeighborPts) >= min_Pts:
                        for a in Ner_NeighborPts:
                            if a not in NeighborPts:
                                NeighborPts.append(a)
                    if (cluster[i]==-1):
                        cluster[i]=k
    return cluster

def generatorData():

    X1, y1=datasets.make_circles(n_samples=5000, factor=.6,
                                           noise=.05)
    print(X1)
    X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2,1.2]], cluster_std=[[.1]],
                random_state=9)
    print(X2)
    X = np.concatenate((X1, X2))
    return X

import cv2

def images_to_video(path):
    img_array = []
    
    imgList = os.listdir(path)
    imgList.sort(key=lambda x: int(x.split('.')[0])) 
    for count in range(0, len(imgList)): 
        filename = imgList[count]
        img = cv2.imread(path + filename)
        if img is None:
            print(filename + " is error!")
            continue
        img_array.append(img)

    height, width, layers = img.shape
    size = (width, height)
    fps = 5  # 设置每帧图像切换的速度
    out = cv2.VideoWriter('demo-jitter.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
 
def main():
    path = "img/"  # 改成你自己图片文件夹的路径
    images_to_video(path)
 


if __name__=='__main__':
    db={}
    eps=0.08
    min_Pts=10
    begin=time.time()
    datapath='TrajectoryData_students003/TrajectoryData_students003/students003.txt'
    data=open(datapath,'r',encoding='utf-8',errors='ignore')
    line=data.readline()
    step=[]
    ID=[]
    ID1=[]
    X=[]
    Y=[]
    relation=[[0]*435 for i in range(435)]
    t_relation={}
    t_group={}
    x1={}
    y1={}
    x2={}
    y2={}
    core={}
    index=10
    while line:
        m=line.split()
        m=list(map(eval,m))
        a=m[0:1]
        step.append(a)
        b=m[1:2]
        ID1.append(b)
        c=m[2:3]
        X.append(c)
        d=m[3:4]
        Y.append(d)
        line=data.readline()
    data.close()
    step=np.array(step)
    ID=np.array(ID1)
    X=np.array(X)
    Y=np.array(Y)
    x=np.hstack((X,Y))
    k=0
    a=0
    t=0
    sum=0
    print(x)
    print(len(step))
    for i in range(len(step)):
        if int(step[i])==index:
            xx=[]
            xx=x[k:i]
            C = DBSCAN(eps=1.2, min_samples=1).fit(xx)
            db[index-10]=C.labels_
            core[index-10]=C.core_sample_indices_
            text=[]
            text=ID1[k:i]
            plt.figure(figsize=(12, 9), dpi=80)
            plt.scatter(xx[:,0],xx[:,1],c=C.fit_predict(xx))
            for j in range(len(text)):
                plt.annotate(text[j],xy=(xx[j,0],xx[j,1]),xytext=(xx[j,0]+0.1,xx[j,1]+0.1))
            plt.savefig('img/'+str(t)+'.png')
            plt.clf()
            plt.close()
            sum+=len(db[index-10])
            index+=10
            k=i
            t+=1
    main()
 
