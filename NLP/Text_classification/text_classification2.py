import argparse
from genericpath import samefile
import tarfile
from nltk import probability
from nltk.sem.logic import read_type
from nltk.util import pr
from numpy.random.mtrand import sample
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords as pw
import numpy as np
from numpy import *
import math
import operator
from operator import itemgetter, truth
from numpy import linalg

def DataSet(path,targetname,targetname1):
    # path="20_newsgroups/"
    srcData=os.listdir(path)
    for i in range(len(srcData)):
        srcDatadir=path+srcData[i]
        srcDataList=os.listdir(srcDatadir)
        targetdir=targetname+"/preprocess"+srcData[i]
        targetdir1=targetname1+"/"+srcData[i]
        if not os.path.exists(targetdir):
            os.makedirs(targetdir)
        else:
            print('Exist')
        if not os.path.exists(targetdir1):
            os.makedirs(targetdir1)
        else:
            print('Exist')
        for j in range(len(srcDataList)):
            targetfile=targetname+'/preprocess'+srcData[i]+'/'+srcDataList[j]
            targetfile1=targetname1+'/'+srcData[i]+'/'+srcDataList[j]
            srcfile=path+srcData[i]+'/'+srcDataList[j]
            tarfiles=open(targetfile,'w',encoding='utf-8',errors='ignore')
            tarfiles1=open(targetfile1,'w',encoding='utf-8',errors='ignore')
            datafiles=open(srcfile,'r',encoding='utf-8',errors='ignore').readlines()
            for line in datafiles:
                dataline=preprocess1(line)
                for word in dataline:
                    tarfiles.write('%s\n'%word)
                    tarfiles1.write('%d\t%s\n'%(i,word))
            tarfiles.close()
            tarfiles1.close()
            print('%s%s'%(srcData[i],srcDataList[j]))

def DataSet1(path,targetname):
    # path="20_newsgroups/"
    srcData=os.listdir(path)
    for k in range(len(srcData)):
        f1=path+srcData[k]
        f2=os.listdir(f1)
        tar=targetname+'/preprocess'+srcData[k]
        for i in range(len(f2)):
            srcDatadir=f1+'/'+f2[i]
            srcDataList=os.listdir(srcDatadir)
            targetdir=tar+'/preprocess'+f2[i]
            if not os.path.exists(targetdir):
                os.makedirs(targetdir)
            else:
                print('Exist')
            for j in range(len(srcDataList)):
                targetfile=targetdir+'/preprocess'+srcDataList[j]
                srcfile=srcDatadir+'/'+srcDataList[j]
                tarfiles=open(targetfile,'w',encoding='utf-8',errors='ignore')
                datafiles=open(srcfile,'r',encoding='utf-8',errors='ignore').readlines()
                for line in datafiles:
                    dataline=preprocess1(line)
                    for word in dataline:
                        tarfiles.write('%s\n'%word)
                tarfiles.close()
                print('%s%s'%(f2[i],srcDataList[j]))

def stpList(filepath):
    stopwords=[line.split() for line in open(filepath,'r',encoding='utf-8',errors='ignore').readlines()]
    return stopwords

def preprocess1(line):
    stopword=stpList('stopword.txt')
    cigan=nltk.PorterStemmer()
    splits=re.compile('[^a-zA-Z]')
    word=[cigan.stem(word.lower()) for word in splits.split(line) if len(word)>0 and word.lower() not in stopword]
    word= [x for x in word if x!=' ']
    return word

def IDF(filedir,filename):
    # filedir='preprocess'
    word_doc={}
    word_IDF={}
    tf={}
    doc_num=0.0
    word_List=os.listdir(filedir)
    for k in range(len(word_List)):
        f1=filedir+word_List[k]
        f2=os.listdir(f1)
        for i in range(len(f2)):
            Sampledir=f1+'/'+f2[i]
            Samplefile=os.listdir(Sampledir)
            for j in range(len(Samplefile)):
                word_in_doc_num={}
                word_in_doc_sum=0
                sample=Sampledir+'/'+Samplefile[j]
                files=open(sample,'r',encoding='utf-8',errors='ignore').readlines()
                for line in files:
                    word_in_doc_sum+=1
                    word=line.strip('\n')
                    word_in_doc_num[word]=word_in_doc_num.get(word,0)+1
                    if word in word_doc.keys():
                        word_doc[word].add(Samplefile[j])
                    else:
                        word_doc.setdefault(word,set())
                        word_doc[word].add(Samplefile[j])
                for word,num in word_in_doc_num.items():
                    tf[word]=float(num)/float(word_in_doc_sum)
            print('finished ',i)
    for i in word_doc.keys():
        doc_num=len(word_doc[i])
        IDF=math.log(20017/doc_num)/math.log(10)
        word_IDF[i]=IDF
    IDFbook=open(filename,'w',encoding='utf-8',errors='ignore')
    for word,IDFs in word_IDF.items():
        IDFbook.write('%s %.6f\n'%(word,IDFs))
    IDFbook.close()
    return word_IDF

def train_test(index0,right,trainpercent=0.9):
    fr=open(right,'w',encoding='utf-8',errors='ignore')
    fileDir = '20_newsgroups'
    featureList=os.listdir(fileDir)
    for i in range(len(featureList)):
        sampledir=fileDir+'/'+featureList[i]
        samplefile=os.listdir(sampledir)
        test_index0=index0*(len(samplefile)*(1-trainpercent))
        test_indexn=(index0+1)*(len(samplefile)*(1-trainpercent))
        for j in range(len(samplefile)):
            if(j>=test_index0)and(j<=test_indexn):
                fr.write('%s %s\n'%(samplefile[j],featureList[i]))
                targetdir='Test_Sample/TestSample'+str(index0)+'/'+featureList[i]
            else:
                targetdir='Train_Sample/TrainSample'+str(index0)+'/'+featureList[i]
            if not os.path.exists(targetdir):
                os.makedirs(targetdir)
            samdir=sampledir+'/'+samplefile[j]
            sam=open(samdir,'r',encoding='utf-8',errors='ignore').readlines()
            samw=open(targetdir+'/'+samplefile[j],'w',encoding='utf-8',errors='ignore')
            for line in sam:
                samw.write('%s\n'%line.strip('\n'))
    samw.close()

# DataSet1('Test_Sample/','preprocess_test')
# DataSet1('Train_Sample/','preprocess_train')
# IDF('preprocess_test/','test_idf')
# IDF('preprocess_train/','train_idf')
def word2vec(trainsamdir,feature_train):
    IDF_data={}
    wordsum_test=0.0
    wordsum_train=0.0
    prob=0.0
    condition={}
    xianyan={}
    tf_idf={}
    # trainsamdir='preprocess_train/'
    # testsamdir='preprocess_test/'
    # testidfdir='Test_IDF'
    trainidfdir='Train_IDF'
    # DataSet1(testsamdir,'preprocess_test')
    # DataSet1(trainsamdir,'preprocess_train')
    # IDF(testsamdir,'test_idf')
    fr_test=open('test_idf','r',encoding='utf-8',errors='ignore').readlines()
    IDF_test={}
    for line in fr_test:
        v=line.strip('\n').split(' ')
        word=v[0]
        IDFs=v[1]
        wordsum_test+=1
        IDF_test[word]=IDFs
    # IDF(trainsamdir,'train_idf')
    fr_train=open('train_idf','r',encoding='utf-8',errors='ignore').readlines()
    IDF_train={}
    for line in fr_train:
        v1=line.strip('\n').split(' ')
        word=v1[0]
        IDFs1=v1[1]
        wordsum_train+=1
        IDF_train[word]=IDFs1
    Trainw = open(trainidfdir, 'w',encoding='utf-8',errors='ignore')
    
    feature_train=os.listdir(trainsamdir)
    for i in range(len(feature_train)):
        sampledir2=trainsamdir+'/'+feature_train[i]
        samplefile2=os.listdir(sampledir2)
        for j in range(len(samplefile2)):
            word_in_doc_num1={}
            word_in_doc_sum1=0
            samples1=sampledir2+'/'+samplefile2[j]
            fr1=open(samples1,'r',encoding='utf-8',errors='ignore').readlines()
            for line in fr1:
                word_in_doc_sum1+=1
                word1=line.strip('\n')#.split(' ')
                word_in_doc_num1[word1]=word_in_doc_num1.get(word1,0)+1
            for word,num in word_in_doc_num1.items():
                # word=word.split(' ')
                TF=float(num)/float(word_in_doc_sum1)
                # tf_idf[word]=TF*float(IDF_train.get(word,0))
                condition[word]=(float(num)+0.0001)/(float(word_in_doc_sum1)+wordsum_train)
                xianyan[word]=(float(word_in_doc_sum1))/float(wordsum_train)
                prob=float(condition[word]/(20*xianyan[word]))
                Trainw.write('%d %f %d %s %s %s'%(float(num),prob,float(word_in_doc_sum1),feature_train[i],samplefile2[j],word))
                Trainw.write('\n')
                return prob
            print('finishtrain',j)
    Trainw.close()

def wordnum(dir):
    word_doc_num={}
    word_doc_prob={}
    attdir=os.listdir(dir)
    for i in range(len(attdir)):
        cnt=0
        filedir=dir+'/'+attdir[i]
        file=os.listdir(filedir)
        for j in range(len(file)):
            samfile=filedir+'/'+file[j]
            word=open(samfile,'r',encoding='utf-8',errors='ignore').readlines()
            for k in word:
                cnt+=1
                words=k.strip('\n')
                name=attdir[i]+'_'+words
                word_doc_prob[name]=word_doc_prob.get(name,0)+1
        word_doc_num[attdir[i]]=cnt
    return word_doc_prob,word_doc_num

def wordprob(trainpath,testword,wordnum,wordsum,word_doc_prob):
    prob=0.0
    word_doc_num=wordnum[trainpath]
    for i in range(len(testword)):
        name=trainpath+'_'+testword[i]
        if name in word_doc_prob:
            condition_fenzi=word_doc_prob[name]+1
        else:
            condition_fenzi=0.0+1
        for j in range(20):
            word_doc_num1=float(wordnum.get(j,0))+float(''.join('%s' %id for id in wordsum))
            xianyan=math.log(condition_fenzi/word_doc_num1)
        prob=prob+xianyan
        probabilitys=prob+math.log(word_doc_num)-math.log(float(''.join('%s' %id for id in wordsum)))
    return probabilitys

def BYS(trainpath,testpath,result):
    resultw=open(result,'w',encoding='utf-8',errors='ignore')
    word_doc_prob,word_doc_num=wordnum(trainpath)
    wordsum=sum(word_doc_num.values())
    testfile=os.listdir(testpath)
    for i in range(len(testfile)):
        testdir=testpath+'/'+testfile[i]
        testsam=os.listdir(testdir)
        for j in range(len(testsam)):
            testword=[]
            testsamdir=testdir+'/'+testsam[j]
            lines=open(testsamdir,'r',encoding='utf-8',errors='ignore').readlines()
            for line in lines:
                word=line.strip('\n')
                testword.append(word)
            Pmax=0.0
            traindir=os.listdir(trainpath)
            for k in range(len(traindir)):
                P=wordprob(traindir[k],testword,word_doc_num,wordsum,word_doc_prob)
                if k==0:
                    Pmax=P
                    attribute=traindir[k]
                    continue
                if P>Pmax:
                    Pmax=P
                    attribute=traindir[k]
            resultw.write('%s %s\n'%(testsam[j],attribute))
    resultw.close()

# def BYS(trainpath,testword):  
#     testsamdir='preprocess_test/'
#     testidfdir='Test_IDF'
#     # trainidfdir='Train_IDF' 
#     # Testw = open(testidfdir, 'w',encoding='utf-8',errors='ignore')
#     feature_test=os.listdir(testsamdir)
#     resultw=open('result','w',encoding='utf-8',errors='ignore')
#     for k in range(len(feature_test)):
#         f1=testsamdir+feature_test[k]
#         f2=os.listdir(f1)
#         for i in range(len(f2)):
#             sampledir1=f1+'/'+f2[i]
#             samplefile1=os.listdir(sampledir1)
#             for j in range(len(samplefile1)):
#                 samples=sampledir1+'/'+samplefile1[j]
#                 testword=[]
#                 fr2=open(samples,'r',encoding='utf-8',errors='ignore').readlines()
#                 for line in fr2:
#                     word=line.strip('\n')
#                     testword.append(word)
#                 Pmax=0.0
#                 trainfile=os.listdir(trainpath)
#                 for x in range(len(trainfile)):
#                     P=word2vec(trainfile[x],testword)
#                     if x==0:
#                         Pmax=P
#                         attribute=trainfile[x]
#                         continue
#                     if P>Pmax:
#                         Pmax=P
#                         attribute=trainfile[x]
#                 resultw.write('%s %s\n'%(samplefile1[j],attribute))
#                 print('finish',j)
#     resultw.close()



    
    
    
    # testfile=os.listdir(testsamdir)
    # for i in range(len(testfile)):
    #     testsampledir=testsamdir+testfile[i]
    #     testsamplefile=open(testidfdir,'r',encoding='utf-8',errors='ignore').readlines()
    #     for j in range(len(testsamplefile)):
    #         testword=[]
    #         # samdir=testsampledir+'/'+testsamplefile[j]
    #         # lines=open(samdir,'r',encoding='utf-8',errors='ignore').readlines()
    #         for k in testsamplefile:
    #             words=k.strip('\n')
    #             testword.append(words)
    #         IDFmax=0.0
    #         trainsamplefile=os.listdir(trainsamdir)
    #         # trainsamplefile = open(trainidfdir, 'w',encoding='utf-8',errors='ignore')
    #         for k in range(len(trainsamplefile)):
    #             P=prob.get(k,0)
    #             if k==0:
    #                 IDFmax=P
    #                 attribute=trainsamplefile[k]
    #             if P>IDFmax:
    #                 IDFmax=P
    #                 attribute=trainsamplefile[k]
    #         resultw.write('%s %s %f\n'%(testsamplefile[i],attribute,IDFmax))
    # resultw.close()

def accuracy(right,result,k):
    rightdict={}
    resultdict={}
    rightcnt=0.0
    for i in open(right,'r',encoding='utf-8',errors='ignore').readlines():
        (file,attribute)=i.strip('\n').split(' ')
        rightdict[file]=attribute
    for i in open(result,'r',encoding='utf-8',errors='ignore').readlines():
        (file,attribute)=i.strip('\n').split(' ')
        resultdict[file]=attribute
    for file in rightdict.keys():
        if(rightdict.get(file,0)==resultdict.get(file,0)):
            rightcnt+=1.0
    print('rightcnt: %d right: %d'%(rightcnt,len(rightdict)))
    acc=rightcnt/len(rightdict)
    print('accuracy %d : %f'%(k,acc))
    return acc
                


            
# def sigmod(x):
#     return 1/(1+np.exp(-x))

# def sigmod_dao(x):
#     fx=sigmod(x)
#     return fx*(1-fx)

# def loss(truth,predict):
#     return ((truth-predict)**2).mean()

# class Net:
#     def __init__(self) -> None:
#         self.w1=np.random.normal()
#         self.w2=np.random.normal()
#         self.w3=np.random.normal()
#         self.w4=np.random.normal()
#         self.w5=np.random.normal()
#         self.w6=np.random.normal()
        
#         self.b1=np.random.normal()
#         self.b2=np.random.normal()
#         self.b3=np.random.normal()
    
#     def forward(self,x):
#         h1=sigmod(self.w1*x[0]+self.w2*x[1]+self.b1)
#         h2=sigmod(self.w3*x[0]+self.w4*x[1]+self.b2)
#         out1=sigmod(self.w5*h1+self.w6*h2+self.b3)
#         return out1

#     def train(self,data):
#         learn_rate=0.1
#         epochs=1000
#         for i in range(epochs):
#             for x,truth in data:
#                 h1sum=self.w1*x[0]+self.w2*x[1]+self.b1
#                 h1=sigmod(h1sum)
#                 h2sum=self.w3*x[0]+self.w4*x[1]+self.b2
#                 h2=sigmod(h2sum)
#                 out1sum=self.w5*h1+self.w6*h2+self.b3
#                 out1=sigmod(out1sum)
#                 predict=out1

#                 dis=-2*(truth-predict)
#                 predict_w5=h1*sigmod_dao(out1sum)
#                 predict_w6=h2*sigmod_dao(out1sum)
#                 predict_b3=sigmod_dao(out1sum)
#                 predict_h1=self.w5*sigmod_dao(out1sum)
#                 predict_h2=self.w6*sigmod_dao(out1sum)

#                 h1_w1=x[0]*sigmod_dao(h1sum)
#                 h1_w2=x[1]*sigmod_dao(h1sum)
#                 h1_b1=sigmod_dao(h1sum)

#                 h2_w3=x[0]*sigmod_dao(h2sum)
#                 h2_w4=x[1]*sigmod_dao(h2sum)
#                 h2_b2=sigmod_dao(h2sum)

#                 self.w1-=learn_rate*dis*predict_h1*h1_w1
#                 self.w2-=learn_rate*dis*predict_h1*h1_w2
#                 self.b1-=learn_rate*dis*predict_h1*h1_b1
#                 self.w3-=learn_rate*dis*predict_h2*h2_w3
#                 self.w4-=learn_rate*dis*predict_h2*h2_w4
#                 self.b2-=learn_rate*dis*predict_h2*h2_b2
#                 self.w5-=learn_rate*dis*predict_w5
#                 self.w6-=learn_rate*dis*predict_w6
#                 self.b3-=learn_rate*dis*predict_b3

#             if i%10==0:
#                 predict=np.apply_along_axis(self.forward,1,data)
#                 lossf=loss(truth,predict)
#                 print("Epoch %d loss: %.3f"%(i,lossf))

# data=open('Train_IDF','r',encoding='utf-8',errors='ignore').readlines()

       






def main():
    # DataSet('20_newsgroups/','preprocess','preprocess1')
    # DataSet('preprocess_test/preprocessTestSample0/','preprocess','preprocessT')
    
    # IDF()
    acc=[]
    # for i in range(10):
    #     right='right'+str(i)+'.txt'
    #     train_test(i,right)
    # for i in range(10):
    #     trainpath='Train_Sample/TrainSample'+str(i)
    #     testpath='Test_Sample/TestSample'+str(i)
    #     result='result'+str(i)+'.txt'
    #     BYS(trainpath,testpath,result)
    for i in range(10):
        right1='right'+str(i)+'.txt'
        result='result'+str(i)+'.txt'
        acc.append(accuracy(right1,result,i))
    return acc
    


if __name__ == '__main__':
    main()



