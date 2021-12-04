from logging import _srcfile
import os
import datetime

dictpath='30wChinsesSeqDic_clean.txt'
def IsDict():
    wordic=[]
    datainfo = open(dictpath,'r',encoding='utf-8',errors='ignore').readlines()
    for line in datainfo:
        wordic.append(line.split(" ")[0].strip())
    return wordic  
        
def BMM(sentence,wordic):
    result=[]
    maxlength=max(len(word)for word in wordic)
    sentence=sentence.strip()
    senL=len(sentence)
    while senL>0:
        maxwordlength=min(maxlength,senL)
        subsentence=sentence[-maxwordlength:]
        while maxwordlength>0:
            if subsentence in wordic:
                result.append(subsentence)
                break
            elif maxwordlength==1:
                result.append(subsentence)
                break
            else:
                maxwordlength=maxwordlength-1
                subsentence=subsentence[-maxwordlength:]

        sentence=sentence[0:-maxwordlength]
        senL=senL-maxwordlength
    result.reverse()
    words="/".join(result)               
    return words

def FMM(sentence,wordic):
    result=[]
    maxlength=max(len(word)for word in wordic)
    sentence=sentence.strip()
    senL=len(sentence)
    i=0
    while senL>0:
        maxwordlength=min(maxlength,senL)
        subsentence=sentence[0:maxwordlength]
        while maxwordlength>0:
            if subsentence in wordic:
                result.append(subsentence)
                break
            elif maxwordlength==1:
                result.append(subsentence)
                break
            else:
                maxwordlength=maxwordlength-1
                subsentence=subsentence[0:len(subsentence)-1]
        sentence=sentence[len(subsentence):]
        senL=len(sentence)
    words="/".join(result)
    return words



def main():
    # maxwordlength=5
    sentence="滤波要分通道滤波，这里由于博客问题，显示灰色（好像因为今天清明节）（逝者已矣 生者如斯）"
    wordic=IsDict()
    print('BMM: ')
    result=BMM(sentence,wordic)
    print(result)
    print('\n FMM: ')
    result=FMM(sentence,wordic)
    print(result)

if __name__=='__main__':
    main()
        
    

