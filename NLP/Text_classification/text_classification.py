import os
import numpy as np
from numpy import *
from os import listdir,mkdir,path
import re
from nltk.corpus import stopwords as pw
import nltk
import time
import math
from math import log
import operator
from numba import jit
from operator import itemgetter
from numpy import linalg
nltk.data.path.append("/home/lujiayuan/nltk_data/stopwords")
# path='/home/lujiayuan/python'
# import sys
# sys.path.append('/home/lujiayuan')
##############################################################
## 1. 创建新文件夹，存放预处理后的文本数据
##############################################################
@jit
def createFiles():
    # srcFilesList = os.listdir('/home/lujiayuan/python/20_newsgroups')
    path="/home/lujiayuan/python/20_newsgroups/"
    srcFilesList=os.listdir(path)
    for i in range(len(srcFilesList)):
    # for fpath,dnames,fnames in os.walk(path):
        # print('/home/lujiayuan/python/20_newsgroups'+"\\"+i)
        # if i==0: continue
        dataFilesDir = "/home/lujiayuan/python/20_newsgroups/" + srcFilesList[i] # 20个文件夹每个的路径
        # dataFilesDir=i
        dataFilesList = os.listdir(dataFilesDir)
        targetDir = "preprocess/preprocess" + srcFilesList[i] # 20个新文件夹每个的路径
        if os.path.exists(targetDir)==False:
            os.makedirs(targetDir)
        else:
            print  ('%s exists' % targetDir)
        for j in range(len(dataFilesList)):
            createProcessFile(srcFilesList[i],dataFilesList[j]) # 调用createProcessFile()在新文档中处理文本
            print ('%s %s' % (srcFilesList[i],dataFilesList[j]))
##############################################################
## 2. 建立目标文件夹，生成目标文件
## @param srcFilesName 某组新闻文件夹的文件名，比如alt.atheism
## @param dataFilesName 文件夹下某个数据文件的文件名
## @param dataList 数据文件按行读取后的字符串列表
##############################################################
def createProcessFile(srcFilesName,dataFilesName):
    srcFile = '/home/lujiayuan/python/20_newsgroups/' + srcFilesName + '/' + dataFilesName
    targetFile= 'preprocess/preprocess' + srcFilesName\
                + '/' + dataFilesName
    fw = open(targetFile,'w',encoding='utf-8',errors='ignore')
    dataList = open(srcFile,'r',encoding='utf-8',errors='ignore').readlines()
    for line in dataList:
        resLine = lineProcess(line) # 调用lineProcess()处理每行文本
        for word in resLine:
            fw.write('%s\n' % word) #一行一个单词
    fw.close()
##############################################################
##3. 对每行字符串进行处理，主要是去除非字母字符，转换大写为小写，去除停用词
## @param line 待处理的一行字符串
## @return words 按非字母分隔后的单词所组成的列表
##############################################################
def stpList(filepath):
    stopwords=[line.split() for line in open(filepath,'r',encoding='utf-8',errors='ignore').readlines()]
    return stopwords

def lineProcess(line):
    # stopwords =pw.words('/home/lujiayuan/nltk_data/stopwords/english') #去停用词
    stopwords=stpList('/home/lujiayuan/c++/stopword.txt')
    porter = nltk.PorterStemmer()  #词干分析
    splitter = re.compile('[^a-zA-Z]')  #去除非字母字符，形成分隔
    words = [porter.stem(word.lower()) for word in splitter.split(line)\
            if len(word)>0 and\
            word.lower() not in stopwords]
    return words

#3. 构造字典sortedNewWordMap

########################################################
## 统计每个词的总的出现次数
## @param strDir
## @param wordMap
## return newWordMap 返回字典，<key, value>结构，按key排序，value都大于4，即都是出现次数大于4的词
#########################################################
def countWords():
    wordMap = {}
    newWordMap = {}
    fileDir = './preprocess'
    sampleFilesList = os.listdir(fileDir)
    for i in range(len(sampleFilesList)):
        sampleFilesDir = fileDir + '/' + sampleFilesList[i]
        sampleList = os.listdir(sampleFilesDir)
        for j in range(len(sampleList)):
            sampleDir = sampleFilesDir + '/' + sampleList[j]
            for line in open(sampleDir,'r',encoding='utf-8',errors='ignore').readlines():
                word = line.strip('\n')
                wordMap[word] = wordMap.get(word,0.0) + 1.0
    #只返回出现次数大于4的单词
    for key, value in wordMap.items():
        if value > 4:
            newWordMap[key] = value
    # sortedNewWordMap = sorted(newWordMap.iteritems())
    sortedNewWordMap = sorted(newWordMap.items())
    print ('wordMap size : %d' % len(wordMap))
    print ('newWordMap size : %d' % len(sortedNewWordMap))
    return sortedNewWordMap
############################################################
##打印属性字典
###########################################################
def printWordMap():
    print ('Print Word Map')
    countLine=0
    fr = open('./attribute','w',encoding='utf-8',errors='ignore')
    sortedWordMap = countWords()
    for item in sortedWordMap:
        fr.write('%s %.1f\n' % (item[0],item[1]))
        countLine += 1
    print ('sortedWordMap size : %d' % countLine)


#####################################################
##特征词选取
####################################################
def filterSpecialWords():
    fileDir = './preprocess'
    wordMapDict = {}
    sortedWordMap = countWords()
    for i in range(len(sortedWordMap)):
        wordMapDict[sortedWordMap[i][0]]=sortedWordMap[i][0]
    sampleDir = os.listdir(fileDir)
    for i in range(len(sampleDir)):
        targetDir = './new_preprocess' + '/' + sampleDir[i]
        srcDir = './preprocess' + '/' + sampleDir[i]
        if path.exists(targetDir) == False:
            os.makedirs(targetDir)
        sample = os.listdir(srcDir)
        for j in range(len(sample)):
            targetSampleFile = targetDir + '/' + sample[j]
            fr=open(targetSampleFile,'w',encoding='utf-8',errors='ignore')
            srcSampleFile = srcDir + '/' + sample[j]
            for line in open(srcSampleFile,'r',encoding='utf-8',errors='ignore').readlines():
                word = line.strip('\n')
                if word in wordMapDict.keys():
                    fr.write('%s\n' % word)
            fr.close()

def computeIDF():
    fileDir = './new_preprocess'
    wordDocMap = {}  # <word, set(docM,...,docN)>
    IDFPerWordMap = {}  # <word, IDF值>
    countDoc = 0.0
    cateList = os.listdir(fileDir)
    for i in range(len(cateList)):
        sampleDir = fileDir + '/' + cateList[i]
        sampleList = os.listdir(sampleDir)
        for j in range(len(sampleList)):
            sample = sampleDir + '/' + sampleList[j]
            for line in open(sample,'r',encoding='utf-8',errors='ignore').readlines():
                word = line.strip('\n')
                if word in wordDocMap.keys():
                    wordDocMap[word].add(sampleList[j]) # set结构保存单词word出现过的文档
                else:
                    wordDocMap.setdefault(word,set())
                    wordDocMap[word].add(sampleList[j])
        print ('just finished %d round ' % i)

    for word in wordDocMap.keys():
        countDoc = len(wordDocMap[word]) # 统计set中的文档个数
        IDF = log(20000/countDoc)/log(10)
        IDFPerWordMap[word] = IDF
 
    return IDFPerWordMap

###################################################
## 将IDF值写入文件保存
###################################################    
def IDFbook():
    start=time.time()
    IDFPerWordMap = computeIDF()
    end=time.time()
    print ('runtime: ' + str(end-start))
    fw = open('IDFPerWord','w',encoding='utf-8',errors='ignore')
    for word, IDF in IDFPerWordMap.items():
        fw.write('%s %.6f\n' % (word,IDF))
    fw.close()

##########################################################
## 创建训练样例集合和测试样例集合
## @param indexOfSample 第k次实验
## @param classifyRightCate 第k次实验的测试集中，<doc rightCategory>数据
## @param trainSamplePercent 训练集与测试集的分割比例
############################################################
def createTestSample(indexOfSample,classifyRightCate,trainSamplePercent=0.9):
    fr = open(classifyRightCate,'w',encoding='utf-8',errors='ignore')
    # IDFPerWord = {} # <word, IDF值> 从文件中读入后的数据保存在此字典结构中
    # for line in open('IDFPerWord','r',encoding='utf-8',errors='ignore').readlines():
    #     (word, IDF) = line.strip('\n').split(' ')
    #     IDFPerWord[word] = IDF        
    
    # fileDir1 = '20_newsgroups'
    # trainFileDir = "docVector/TrainFile/" + 'wordTFIDFMapTrainSample' + str(indexOfSample)
    # testFileDir = "docVector/TestFile/" + 'wordTFIDFMapTestSample' + str(indexOfSample)

    # tsTrainWriter = open(trainFileDir, 'w',encoding='utf-8',errors='ignore')
    # tsTestWriter = open(testFileDir, 'w',encoding='utf-8',errors='ignore')
    fileDir = './preprocess'
    sampleFilesList=os.listdir(fileDir)
    for i in range(len(sampleFilesList)):
        sampleFilesDir = fileDir + '/' + sampleFilesList[i]
        sampleList = os.listdir(sampleFilesDir)
        m = len(sampleList)
        testBeginIndex = indexOfSample * ( m * (1-trainSamplePercent) )
        testEndIndex = (indexOfSample + 1) * ( m * (1-trainSamplePercent) )
        for j in range(m):
            # 序号在规定区间内的作为测试样本，需要为测试样本生成类别-序号文件，最后加入分类的结果，
            # 一行对应一个文件，方便统计准确率  
            if (j > testBeginIndex) and (j < testEndIndex): 
                fr.write('%s %s\n' % (sampleList[j],sampleFilesList[i])) # 写入内容：每篇文档序号 它所在的文档名称即分类
                targetDir = './TestSample/TestSample'+str(indexOfSample)+'/'+sampleFilesList[i]
            else:
                targetDir = './TrainSample/TrainSample'+str(indexOfSample)+'/'+sampleFilesList[i]
            if os.path.exists(targetDir) == False:
                os.makedirs(targetDir)
            sampleDir = sampleFilesDir + '/' + sampleList[j]
            sample = open(sampleDir,'r',encoding='utf-8',errors='ignore').readlines()
            sampleWriter = open(targetDir+'/'+sampleList[j],'w',encoding='utf-8',errors='ignore')
            for line in sample:
                sampleWriter.write('%s\n' % line.strip('\n'))
            sampleWriter.close()
    fr.close()
    
# 调用以上函数生成标注集，训练和测试集合
def test():
    for i in range(10):
        classifyRightCate = 'classifyRightCate/classifyRightCate' + str(i) + '.txt'
        createTestSample(i,classifyRightCate)



########################################################################
## 统计训练样本中，每个目录下每个单词的出现次数, 及每个目录下的单词总数
## @param 训练样本集目录
## @return cateWordsProb <类目_单词 ,某单词出现次数>
## @return cateWordsNum <类目，单词总数>
#########################################################################
def getCateWordsProb(strDir):
    #strDir = TrainSample0
    cateWordsNum = {}
    cateWordsProb = {}
    cateDir = os.listdir(strDir)
    for i in range(len(cateDir)):
        count = 0 # 记录每个目录下（即每个类下）单词总数
        sampleDir = strDir + '/' + cateDir[i]
        sample = os.listdir(sampleDir)
        for j in range(len(sample)):
            sampleFile = sampleDir + '/' + sample[j]
            words = open(sampleFile,'r',encoding='utf-8',errors='ignore').readlines()
            for line in words:
                count = count + 1
                word = line.strip('\n')
                keyName = cateDir[i] + '_' + word
                cateWordsProb[keyName] = cateWordsProb.get(keyName,0)+1 # 记录每个目录下（即每个类下）每个单词的出现次数
        cateWordsNum[cateDir[i]] = count
        print ('cate %d contains %d' % (i,cateWordsNum[cateDir[i]]))
    print ('cate-word size: %d' % len(cateWordsProb))
    return cateWordsProb, cateWordsNum



##########################################
## 用贝叶斯对测试文档分类
## @param traindir 训练集目录
## @param testdir  测试集目录
## @param classifyResultFileNew  分类结果文件
## @return 返回该测试样本在该类别的概率
##########################################
def NBprocess(traindir,testdir,classifyResultFileNew):
    crWriter = open(classifyResultFileNew,'w',encoding='utf-8',errors='ignore')
    # traindir = 'TrainSample0'
    # testdir = 'TestSample0'
    #返回类k下词C的出现次数，类k总词数
    cateWordsProb, cateWordsNum = getCateWordsProb(traindir)
    
    #训练集的总词数
    trainTotalNum = sum(cateWordsNum.values())
    print ('trainTotalNum: %s' % trainTotalNum)
    
    #开始对测试样例做分类
    testDirFiles = os.listdir(testdir)
    for i in range(len(testDirFiles)):
        testSampleDir = testdir + '/' + testDirFiles[i]
        testSample = os.listdir(testSampleDir)
        for j in range(len(testSample)):
            testFilesWords = []
            sampleDir = testSampleDir + '/' + testSample[j]
            lines = open(sampleDir,'r',encoding='utf-8',errors='ignore').readlines()
            for line in lines:
                word = line.strip('\n')
                testFilesWords.append(word)
    
            maxP = 0.0
            trainDirFiles = os.listdir(traindir)
            for k in range(len(trainDirFiles)):
                p = computeCateProb(trainDirFiles[k], testFilesWords,\
                                    cateWordsNum, trainTotalNum, cateWordsProb)
                if k==0:
                    maxP = p
                    bestCate = trainDirFiles[k]
                    continue
                if p > maxP:
                    maxP = p
                    bestCate = trainDirFiles[k]
            crWriter.write('%s %s\n' % (testSample[j],bestCate))
    crWriter.close()
    
#################################################
## @param traindir       类k
## @param testFilesWords 某个测试文档
## @param cateWordsNum   训练集类k下单词总数 <类目，单词总数>
## @param totalWordsNum  训练集单词总数
## @param cateWordsProb  训练集类k下词c出现的次数 <类目_单词 ,某单词出现次数>
## 计算 条件概率 =（类k中单词i的数目+0.0001）/（类k中单词总数+训练样本中所有类单词总数）
## 计算 先验概率 =（类k中单词总数）/（训练样本中所有类单词总数）
#################################################
def computeCateProb(traindir,testFilesWords,cateWordsNum,\
                    totalWordsNum,cateWordsProb):
    prob = 0
    wordNumInCate = cateWordsNum[traindir]  # 类k下单词总数 <类目，单词总数>
    # a=float(''.join('%s' %id for id in wordNumInCate))+float(''.join('%s' %id for id in totalWordsNum))
    for i in range(len(testFilesWords)):
        keyName = traindir + '_' + testFilesWords[i]
        # if cateWordsProb.has_key(keyName):
        # if cateWordsProb.__contains__(keyName):
        if keyName in cateWordsProb:
            testFileWordNumInCate = cateWordsProb[keyName] +0.0001# 类k下词c出现的次数
        else: testFileWordNumInCate = 0.0+0.0001
        # xcProb = math.log((map(float,testFileWordNumInCate) + 0.0001) /# \ # 求对数避免很多很小的数相乘下溢出
        # xcProb=math.log(list((map(lambda x:x+testFileWordNumInCate,0.0001)))/
        # xcProb = math.log(list(testFileWordNumInCate + 0.0001)) /(list(totalWordsNum).append(wordNumInCate))
        for j in range(20):
            wordNumInCate1=float(cateWordsNum.get(j,0))+float(''.join('%s' %id for id in totalWordsNum))
            xcProb=math.log(testFileWordNumInCate/wordNumInCate1)
        # xcProb = math.log((testFileWordNumInCate + 0.0001) /(totalWordsNum+list(wordNumInCate)))
                    # (wordNumInCate + list(totalWordsNum)))
                    # (list(wordNumInCate).append(totalWordsNum)))
                    # (list(map(lambda x:x+wordNumInCate,totalWordsNum)))
                    # (wordNumInCate + list(totalWordsNum)))
        prob = prob + xcProb
        res = prob + math.log(wordNumInCate) - math.log(float(''.join('%s' %id for id in totalWordsNum)))
    return res



def computeAccuracy(rightCate,resultCate,k):
    rightCateDict = {}
    resultCateDict = {}
    rightCount = 0.0
    
    for line in open(rightCate,'r',encoding='utf-8',errors='ignore').readlines():
        (sampleFile,cate) = line.strip('\n').split(' ')
        rightCateDict[sampleFile] = cate
    
    for line in open(resultCate,'r',encoding='utf-8',errors='ignore').readlines():
        (sampleFile,cate) = line.strip('\n').split(' ')
        resultCateDict[sampleFile] = cate
    
    for sampleFile in rightCateDict.keys():
        #print 'rightCate: %s  resultCate: %s' % \
            #     (rightCateDict[sampleFile],resultCateDict[sampleFile])
        #print 'equal or not: %s' % (rightCateDict[sampleFile]==resultCateDict[sampleFile])
    
        if (rightCateDict.get(sampleFile,'20839')==resultCateDict.get(sampleFile,'20839')):
            rightCount += 1.0
    print ('rightcntt : %d  rightattribute: %d' % (rightCount,len(rightCateDict)))
    accuracy = rightCount/len(rightCateDict)
    print ('accuracy %d : %f' % (k,accuracy))
    return accuracy



#############################################################################
## 生成每次迭代的测试用例、标注集
def step1():
    for i in range(10):
        classifyRightCate = 'classifyRightCate' + str(i) + '.txt'
        createTestSample(i,classifyRightCate)
##############################################################################
## bayes对测试文档做分类
def step2():
    for i in range(10):
        traindir = 'TrainSample/TrainSample' + str(i)
        testdir = 'TestSample/TestSample' + str(i)
        classifyResultFileNew = 'classifyResultFileNew' + str(i) + '.txt'
        NBprocess(traindir,testdir,classifyResultFileNew)
##############################################################################
## 计算准确率
def step3():
    accuracyOfEveryExp = []
    for i in range(10):
        rightCate = 'classifyRightCate'+str(i)+'.txt'
        resultCate = 'classifyResultFileNew'+str(i)+'.txt'
        accuracyOfEveryExp.append(computeAccuracy(rightCate,resultCate,i))
    return accuracyOfEveryExp


def computeTFMultiIDF(indexOfSample,trainSamplePercent=0.9):
    IDFPerWord = {} # <word, IDF值> 从文件中读入后的数据保存在此字典结构中
    for line in open('IDFPerWord','r',encoding='utf-8',errors='ignore').readlines():
        (word, IDF) = line.strip('\n').split(' ')
        IDFPerWord[word] = IDF        
    
    fileDir = '20_newsgroups'
    trainFileDir = "docVector/TrainFile/" + 'wordTFIDFMapTrainSample' + str(indexOfSample)
    testFileDir = "docVector/TestFile/" + 'wordTFIDFMapTestSample' + str(indexOfSample)

    tsTrainWriter = open(trainFileDir, 'w',encoding='utf-8',errors='ignore')
    tsTestWriter = open(testFileDir, 'w',encoding='utf-8',errors='ignore')

        
    cateList = os.listdir(fileDir)
    for i in range(len(cateList)):
        sampleDir = fileDir + '/' + cateList[i]
        sampleList = os.listdir(sampleDir)
        
        testBeginIndex = indexOfSample * ( len(sampleList) * (1-trainSamplePercent) )
        testEndIndex = (indexOfSample+1) * ( len(sampleList) * (1-trainSamplePercent) )
        
        for j in range(len(sampleList)):
            TFPerDocMap = {} # <word, 文档doc下该word的出现次数>
            sumPerDoc = 0  # 记录文档doc下的单词总数
            sample = sampleDir + '/' + sampleList[j]
            for line in open(sample,'r',encoding='utf-8',errors='ignore').readlines():
                sumPerDoc += 1
                word = line.strip('\n')
                TFPerDocMap[word] = TFPerDocMap.get(word, 0) + 1
            
            if(j >= testBeginIndex) and (j <= testEndIndex):
                tsWriter = tsTestWriter
            else:
                tsWriter = tsTrainWriter

            tsWriter.write('%s %s ' % (cateList[i], sampleList[j])) # 写入类别cate，文档doc

            for word, count in TFPerDocMap.items():
                TF = float(count)/float(sumPerDoc)
                tsWriter.write('%s %f ' % (word, TF * float(IDFPerWord.get(word,0)))) # 继续写入类别cate下文档doc下的所有单词及它的TF-IDF值

            tsWriter.write('\n')

        print ('just finished %d round ' % i)

        #if i==0: break

    tsTrainWriter.close()
    tsTestWriter.close()
    tsWriter.close()

def test1():
    for i in range(10):
        # classifyRightCate = 'classifyRightCate/classifyRightCate' + str(i) + '.txt'
        computeTFMultiIDF(i)

def doProcess():
    # traindir = './TrainSample/TrainSample0'
    # testdir = './TestSample/TestSample0'
    # kNNResultFile = 'KNNClassifyResult'

    traindir = 'docVector/TrainFile/wordTFIDFMapTrainSample0'
    testdir = 'docVector/TestFile/wordTFIDFMapTestSample0'
    kNNResultFile = 'docVector/KNNClassifyResult'

    trainDocWordMap = {}  # 字典<key, value> key=cate_doc, value={{word1,tfidf1}, {word2, tfidf2},...}
    # trainDirFiles = os.listdir(testdir)
    # for i in range(len(trainDirFiles)):
    for line in open(traindir,'r',encoding='utf-8',errors='ignore').readlines():
        lineSplitBlock = line.strip('\n').split(' ')
        trainWordMap = {}
        m = len(lineSplitBlock)-1
        for i in range(2, m, 2):  # 在每个文档向量中提取(word, tfidf)存入字典
            trainWordMap[lineSplitBlock[i]] = lineSplitBlock[i+1]

        temp_key = lineSplitBlock[0] + '_' + lineSplitBlock[1]  # 在每个文档向量中提取类目cate，文档doc，
        trainDocWordMap[temp_key] = trainWordMap 

    testDocWordMap = {}
    # testDirFiles = os.listdir(testdir)
    # for i in range(len(testDirFiles)):
    for line in open(testdir,'r',encoding='utf-8',errors='ignore').readlines():
        lineSplitBlock = line.strip('\n').split(' ')
        testWordMap = {} 
        m = len(lineSplitBlock)-1
        for i in range(2, m, 2):
            testWordMap[lineSplitBlock[i]] = lineSplitBlock[i+1]

        temp_key = lineSplitBlock[0] + '_' + lineSplitBlock[1]
        testDocWordMap[temp_key] = testWordMap #<类_文件名，<word, TFIDF>>

    
    #遍历每一个测试样例计算与所有训练样本的距离，做分类
    count = 0
    rightCount = 0
    KNNResultWriter = open(kNNResultFile,'w',encoding='utf-8',errors='ignore')
    for item in testDocWordMap.items():
        classifyResult = KNNComputeCate(item[0], item[1], trainDocWordMap)  # 调用KNNComputeCate做分类

        count += 1
        print ('this is %d round' % count)

        classifyRight = item[0].split('_')[0]
        KNNResultWriter.write('%s %s\n' % (classifyRight,classifyResult))
        if classifyRight == classifyResult:
            rightCount += 1
        print ('%s %s rightCount:%d' % (classifyRight,classifyResult,rightCount))

    accuracy = float(rightCount)/float(count)
    print ('rightCount : %d , count : %d , accuracy : %.6f' % (rightCount,count,accuracy))
    return accuracy
            


#########################################################
## @param cate_Doc 测试集<类别_文档>
## @param testDic 测试集{{word, TFIDF}}
## @param trainMap 训练集<类_文件名，<word, TFIDF>>
## @return sortedCateSimMap[0][0] 返回与测试文档向量距离和最小的类
#########################################################
def KNNComputeCate(cate_Doc, testDic, trainMap):
    simMap = {} #<类目_文件名,距离> 后面需要将该HashMap按照value排序
    for item in trainMap.items():
        similarity = computeSim(testDic,item[1])  # 调用computeSim()
        simMap[item[0]] = similarity

    sortedSimMap = sorted(simMap.iteritems(), key=itemgetter(1), reverse=True) #<类目_文件名,距离> 按照value排序

    k = 20
    cateSimMap = {} #<类，距离和>
    for i in range(k):
        cate = sortedSimMap[i][0].split('_')[0]
        cateSimMap[cate] = cateSimMap.get(cate,0) + sortedSimMap[i][1]

    sortedCateSimMap = sorted(cateSimMap.iteritems(),key=itemgetter(1),reverse=True)

    return sortedCateSimMap[0][0]   
        
    
#################################################
## @param testDic 一维测试文档向量<<word, tfidf>>
## @param trainDic 一维训练文档向量<<word, tfidf
## @return 返回余弦相似度
def computeSim(testDic, trainDic):
    testList = []  # 测试向量与训练向量共有的词在测试向量中的tfidf值
    trainList = []  # # 测试向量与训练向量共有的词在训练向量中的tfidf值
    
    for word, weight in testDic.items():
        # lines=testDic.keys().readlines()
        # if trainDic.has_key(word):
        if word in trainDic:
            a=trainDic.get(weight,0).split("!")
            # b=''.join('%s' %id for id in a).split("@")
            # c=''.join('%s' %id for id in b).split(" ")
            testList.append(float(trainDic.get(a,0))) # float()将字符型数据转换成数值型数据，参与下面运算
            # for word in lines:
            #     word=word.split(' ')
            a=trainDic.get(word,0).split(".")
            b=''.join('%s' %id for id in a).split("@")
            c=''.join('%s' %id for id in b).split(" ")
            # # c=''.join('%s' %id for id in b).split("irvine")
            
            # c=''.join('%s' %id for id in b).split("irvineuxhcsouiucedu")
            # d=''.join('%s' %id for id in c).split(" ")
            # # c=b
            # # for i in range(len(b)):
            # #     c[i]=ord(b[i])
            try:
                b = [float(s) for s in b]
                for i, value in enumerate(b):
                    b[i] = float(value)
            except ValueError as F:
                print(F)
            # word = [float(s) for s in word]
            # for i, value in enumerate(word):
            #     word[i] = float(value)

            trainList.append(float(trainDic.get(weight,0))) # float()将字符型数据转换成数值型数据，参与下面运算
            # # for word in lines:
            # trainList.append(float(''.join('%s' %id for id in list(map(float,d)))))
            
    testVect = mat(testList)  # 列表转矩阵，便于下面向量相乘运算和使用Numpy模块的范式函数计算
    trainVect = mat(trainList)
    # num = float(testVect * trainVect.T)
    num = float(testVect * trainVect.T)
    denom = linalg.norm(testVect) * linalg.norm(trainVect)
    #print 'denom:%f' % denom
    return float(num)/(1.0+float(denom))

def main():
    # createFiles()
    # countWords()
    # printWordMap()
    # filterSpecialWords()
    # test()
    # step1()
    # step2()
    step3()
    # IDFbook()
    # test1()
    # doProcess()

if __name__=='__main__':
    main()
