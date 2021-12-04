from sentence_transformers import SentenceTransformer
import sklearn.preprocessing as prepro
import numpy as np


def go(s1: str, s2: str) -> float:
    
    model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')
    model = model.cuda()
    sentence_embeddings2 = model.encode(s2)

    a = prepro.normalize(sentence_embeddings1)
    b = prepro.normalize(sentence_embeddings2)
    inlier_product = a.dot(b.T)
    
    return (inlier_product[0,0])
