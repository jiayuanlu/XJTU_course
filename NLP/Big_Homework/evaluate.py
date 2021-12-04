import utils
import utils1
import utils2
import utils3
import utils4
import config
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
# os.environ['CUDA_VISIBLE_DEVICES']='0'
# CUDA_VISIBLE_DEVICES=0


with open(os.path.join(config.datasetPath, "test.tsv"), "r", encoding='utf8') as f:
    lines = f.readlines()
lines = [line.strip().split('\t') for line in lines]

scores0 = []
label0 = []
text0 = []
length0 = []
# for line in tqdm(lines[1:8606]):
for line in tqdm(lines[1:10000]):
    vec0 = utils.sent2vec(line[0])
    vec1 = utils.sent2vec(line[1])
    score = np.dot(vec0, vec1.T)[0, 0]
    scores0.append(score)
    label0.append(int(line[2]))
    text0.append((line[0], line[1]))
    length0.append((len(line[0]) + len(line[1])) / 2)

scores0 = np.array(scores0)
label0 = np.array(label0).astype(int)

fpr, tpr, thresholds  =  roc_curve(label0, scores0)
roc_auc=auc(fpr,tpr)
print('distiluse-base-multilingual-cased-v1: ',roc_auc)
plt.figure()
lw = 2
plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='distiluse-base-multilingual-cased-v1 (area = %0.6f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
# plt.legend(loc="lower right")
# plt.show()

# plt.hist(scores[label == 0])
# plt.show()
# plt.hist(scores[label == 1])
# plt.show()

scores2 = []
label2 = []
text2 = []
for line in tqdm(lines[1:10000]):
    vec0 = utils2.sent2vec(line[0])
    vec1 = utils2.sent2vec(line[1])
    score2 = np.dot(vec0, vec1.T)[0, 0]
    scores2.append(score2)
    label2.append(int(line[2]))
    text2.append((line[0], line[1]))

scores2 = np.array(scores2)
label2 = np.array(label2).astype(int)

fpr, tpr, thresholds  =  roc_curve(label2, scores2)
roc_auc=auc(fpr,tpr)
print('distiluse-base-multilingual-cased-v2: ',roc_auc)
# plt.figure(0).clf()
lw = 2
# plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='blue',
         lw=lw, label='distiluse-base-multilingual-cased-v2 (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线

scores3 = []
label3 = []
text3 = []
for line in tqdm(lines[1:10000]):
    vec0 = utils3.sent2vec(line[0])
    vec1 = utils3.sent2vec(line[1])
    score3 = np.dot(vec0, vec1.T)[0, 0]
    scores3.append(score3)
    label3.append(int(line[2]))
    text3.append((line[0], line[1]))

scores3 = np.array(scores3)
label3 = np.array(label3).astype(int)

fpr, tpr, thresholds  =  roc_curve(label3, scores3)
roc_auc=auc(fpr,tpr)
print('all-distilroberta-v1: ',roc_auc)
# plt.figure(0).clf()
lw = 2
# plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='green',
         lw=lw, label='all-distilroberta-v1 (area = %0.6f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线


scores1 = []
label1 = []
text1 = []
length1 = []
# for line in tqdm(lines[1:8606]):
for line in tqdm(lines[1:10000]):
    vec0 = utils1.sent2vec(line[0])
    vec1 = utils1.sent2vec(line[1])
    score1 = np.dot(vec0, vec1.T)[0, 0]
    scores1.append(score1)
    label1.append(int(line[2]))
    text1.append((line[0], line[1]))
    length1.append((len(line[0]) + len(line[1])) / 2)

scores1 = np.array(scores1)
label1 = np.array(label1).astype(int)

fpr, tpr, thresholds  =  roc_curve(label1, scores1)
roc_auc=auc(fpr,tpr)
print('all-MiniLM-L6-v2: ',roc_auc)
# plt.figure(0).clf()
lw = 2
# plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='red',
         lw=lw, label='all-MiniLM-L6-v2 (area = %0.6f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC')
# plt.legend(loc="lower right")
# plt.show()

# plt.hist(scores1[label1 == 0])
# plt.show()
# plt.hist(scores1[label1 == 1])
# plt.show()

# scores2 = []
# label2 = []
# text2 = []
# for line in tqdm(lines[1:2000]):
#     vec0 = utils2.sent2vec(line[0])
#     vec1 = utils2.sent2vec(line[1])
#     score2 = np.dot(vec0, vec1.T)[0, 0]
#     scores2.append(score2)
#     label2.append(int(line[2]))
#     text2.append((line[0], line[1]))

# scores2 = np.array(scores2)
# label2 = np.array(label2).astype(int)

# fpr, tpr, thresholds  =  roc_curve(label2, scores2)
# roc_auc=auc(fpr,tpr)
# print('distiluse-base-multilingual-cased-v2: ',roc_auc)
# # plt.figure(0).clf()
# lw = 2
# # plt.figure(figsize=(10,10))
# plt.plot(fpr, tpr, color='blue',
#          lw=lw, label='distiluse-base-multilingual-cased-v2 (area = %0.2f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
# # # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# # # plt.xlim([0.0, 1.0])
# # # plt.ylim([0.0, 1.05])
# # # plt.xlabel('False Positive Rate')
# # # plt.ylabel('True Positive Rate')
# # # plt.title('ROC')
# # # plt.legend(loc="lower right")
# # # plt.show()

# scores3 = []
# label3 = []
# text3 = []
# for line in tqdm(lines[1:2000]):
#     vec0 = utils3.sent2vec(line[0])
#     vec1 = utils3.sent2vec(line[1])
#     score3 = np.dot(vec0, vec1.T)[0, 0]
#     scores3.append(score3)
#     label3.append(int(line[2]))
#     text3.append((line[0], line[1]))

# scores3 = np.array(scores3)
# label3 = np.array(label3).astype(int)

# fpr, tpr, thresholds  =  roc_curve(label3, scores3)
# roc_auc=auc(fpr,tpr)
# print('all-distilroberta-v1: ',roc_auc)
# # plt.figure(0).clf()
# lw = 2
# # plt.figure(figsize=(10,10))
# plt.plot(fpr, tpr, color='green',
#          lw=lw, label='all-distilroberta-v1 (area = %0.6f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
# # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.05])
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('ROC')
# # plt.legend(loc="lower right")
# # plt.show()

scores4 = []
label4 = []
text4 = []
for line in tqdm(lines[1:10000]):
    vec0 = utils4.sent2vec(line[0])
    vec1 = utils4.sent2vec(line[1])
    score4 = np.dot(vec0, vec1.T)[0, 0]
    scores4.append(score4)
    label4.append(int(line[2]))
    text4.append((line[0], line[1]))

scores4 = np.array(scores4)
label4 = np.array(label4).astype(int)

fpr, tpr, thresholds  =  roc_curve(label4, scores4)
roc_auc=auc(fpr,tpr)
print('paraphrase-albert-small-v2: ',roc_auc)
# plt.figure(0).clf()
lw = 2
# plt.figure(figsize=(10,10))
plt.plot(fpr, tpr, color='yellow',
         lw=lw, label='paraphrase-albert-small-v2 (area = %0.6f)' % roc_auc) ###假正率为横坐标，真正率为纵坐标做曲线
# # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.05])
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('ROC')
plt.legend(loc="lower right")
plt.show()


# import utils
# import config
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from sklearn.metrics import auc, roc_curve

# with open(os.path.join(config.datasetPath, "train.tsv"), "r", encoding='utf8') as f:
#     lines = f.readlines()
# lines = [line.strip().split('\t') for line in lines]

# scores = []
# label = []
# text = []
# length = []
# for line in tqdm(lines[1:2000]):
#     vec0 = utils.sent2vec(line[0])
#     vec1 = utils.sent2vec(line[1])
#     score = np.dot(vec0, vec1.T)[0, 0]
#     scores.append(score)
#     label.append(int(line[2]))
#     text.append((line[0], line[1]))
#     length.append((len(line[0]) + len(line[1])) / 2)

# scores = np.array(scores)
# label = np.array(label).astype(int)

# fpr, tpr, thresholds = roc_curve(label, scores)
# print(auc(fpr, tpr))
# plt.plot(fpr, tpr)
# plt.show()

'''
plt.scatter(length, abs(scores - label))
plt.show()
'''