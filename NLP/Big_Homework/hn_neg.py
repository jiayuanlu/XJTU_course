from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, evaluation, losses
from torch.utils.data import DataLoader
import numpy as np
from copy import deepcopy
from random import randint
import time
from tqdm import tqdm


def pretrain_feat(text1, text2):
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    F1_mat = model.encode(text1)
    F2_mat = model.encode(text2)
    np.save('F1_mat', F1_mat)
    np.save('F2_mat', F2_mat)


def shuffle2(text1, text2, K=100):
    temp_lst = []
    s_lst = []
    m = len(text2)
    pretrain_feat(text1, text2)
    F1_mat = np.load('F1_mat.npy')
    F2_mat = np.load('F2_mat.npy')

    for i in tqdm(range(m)):
        idx = list(np.random.choice(m, K, replace=False))
        if i in idx:
            idx.remove(i)
        f1 = F1_mat[i]  # [1,D]
        F2_K = F2_mat[idx]  # [K,D]
        scores = f1.dot(F2_K.T)
        hn_id = np.argmax(scores)
        s_lst.append(scores[hn_id])
        temp_lst.append(text2[hn_id])

    return temp_lst, s_lst

if __name__ == '__main__':

    filename = "train_small.txt"
    with open(filename, "r", encoding='utf8') as f:
        lines = f.readlines()
        lines = [line.strip().split('\t') for line in lines]

    sent_list1 = [line[0] for line in lines if line[2] == '1']  # 所有正样本的第一列
    sent_list2 = [line[1] for line in lines if line[2] == '1']  # 所有正样本的第二列
    train_size = int(len(sent_list1) * 0.8)
    eval_size = len(sent_list1) - train_size

    # shuffle_list2 = shuffle(sent_list1)  # 改变第二列的顺序即可得到负样本
    shuffle_list2, s_list = shuffle2(sent_list1, sent_list2)
    with open('hn_neg.txt', "w", encoding='utf8') as f:
        for i in range(len(sent_list1)):
            f.write(sent_list1[i])
            f.write('\t')
            f.write(shuffle_list2[i])
            f.write('\t')
            f.write(f'{s_list[i]:.2f}')
            f.write('\n')