from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, evaluation, losses
from torch.utils.data import DataLoader
import xlrd
from copy import deepcopy
from random import randint
import time


def shuffle(lst):
    temp_lst = deepcopy(lst)
    m = len(temp_lst)
    while (m):
      m -= 1
      i = randint(0, m)
      temp_lst[m], temp_lst[i] = temp_lst[i], temp_lst[m]
    return temp_lst


def load_data(filename):
    with open(filename, "r", encoding='utf8') as f:
        lines = f.readlines()
        lines = [line.strip().split('\t') for line in lines]
        
    sent_list1 = [line[0] for line in lines if line[2] == '1']  # 所有正样本的第一列
    sent_list2 = [line[1] for line in lines if line[2] == '1']  # 所有正样本的第二列
    train_size = int(len(sent_list1) * 0.8)
    eval_size = len(sent_list1) - train_size

    shuffle_list2 = shuffle(sent_list2)  # 改变第二列的顺序即可得到负样本

    # Define your train examples.
    train_data = []
    for idx in range(train_size):
        train_data.append(InputExample(texts=[sent_list1[idx], sent_list2[idx]], label=1.0))  # 添加正样本
        train_data.append(InputExample(texts=[sent_list1[idx], shuffle_list2[idx]], label=0.0))  # 添加负样本

    # Define your evaluation examples
    eval_sent1 = sent_list1[train_size:]
    eval_sent2 = sent_list2[train_size:]
    eval_sent1.extend(list(sent_list1[train_size:]))
    eval_sent2.extend(list(shuffle_list2[train_size:]))
    scores = [1.0] * eval_size + [0.0] * eval_size

    evaluator = evaluation.EmbeddingSimilarityEvaluator(eval_sent1, eval_sent2, scores)

    return train_data, evaluator


if __name__ == '__main__':
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    tic = time.time()
    train_data, evaluator = load_data("lcqmc/train.tsv")
    print(time.time()-tic)
    # Define your train dataset, the dataloader and the train loss
    train_dataset = SentencesDataset(train_data, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=32)
    train_loss = losses.CosineSimilarityLoss(model)
    # Tune the model
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1,
              warmup_steps=100, evaluator=evaluator, evaluation_steps=100, output_path='./govenModel')




# from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, evaluation, losses
# from torch.utils.data import DataLoader
# from copy import deepcopy
# from random import randint
# import time


# def shuffle(lst):
#     temp_lst = deepcopy(lst)
#     m = len(temp_lst)
#     while (m):
#         m -= 1
#         i = randint(0, m)
#         temp_lst[m], temp_lst[i] = temp_lst[i], temp_lst[m]
#     return temp_lst


# def load_data(filename,filename2):
#     with open(filename, "r", encoding='utf8') as f:
#         lines = f.readlines()
#         lines = [line.strip().split('\t') for line in lines]

#     # pos_list1 = [line[0] for line in lines if line[2] == '1']  # 所有正样本的第一列
#     # pos_list2 = [line[1] for line in lines if line[2] == '1']  # 所有正样本的第二列
#     pos_list1 = [line[0] for line in lines ]  # 所有正样本的第一列
#     pos_list2 = [line[1] for line in lines ]
    
#     with open(filename2, "r", encoding='utf8') as f:
#         lines = f.readlines()
#         lines = [line.strip().split('\t') for line in lines]
    
#     # neg_list1 = [line[0] for line in lines if line[2] == '0']
#     # neg_list2 = [line[1] for line in lines if line[2] == '0']
#     neg_list1 = [line[0] for line in lines ]
#     neg_list2 = [line[1] for line in lines ]

#     train_data = []
#     pos_train_size = int(len(pos_list1) * 0.8)
#     pos_eval_size = len(pos_list1) - pos_train_size
#     for idx in range(pos_train_size):
#         train_data.append(InputExample(texts=[pos_list1[idx], pos_list2[idx]], label=1.0))  # 添加正样本
#     neg_train_size = int(len(neg_list1) * 0.8)
#     neg_eval_size = len(neg_list1) - neg_train_size
#     for idx in range(neg_train_size):
#         train_data.append(InputExample(texts=[neg_list1[idx], neg_list2[idx]], label=0.0))  # 添加负样本

#     # Define your evaluation examples
#     eval_sent1 = pos_list1[pos_train_size:]
#     eval_sent2 = pos_list2[pos_train_size:]
#     eval_sent1.extend(list(neg_list1[neg_train_size:]))
#     eval_sent2.extend(list(neg_list2[neg_train_size:]))
#     scores = [1.0] * pos_eval_size + [0.0] * neg_eval_size

#     evaluator = evaluation.EmbeddingSimilarityEvaluator(eval_sent1, eval_sent2, scores)

#     return train_data, evaluator


# if __name__ == '__main__':
#     model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
#     tic = time.time()
#     train_data, evaluator = load_data("govern_pos_neg/pos.tsv","govern_pos_neg/neg.tsv")
#     print(time.time() - tic)
#     # Define your train dataset, the dataloader and the train loss
#     train_dataset = SentencesDataset(train_data, model)
#     train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
#     train_loss = losses.CosineSimilarityLoss(model)
#     # Tune the model
#     model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1,
#               warmup_steps=100, evaluator=evaluator, evaluation_steps=100, output_path='./govenModel1')
