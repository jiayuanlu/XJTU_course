# import numpy as np

# import config
# import utils
# from sklearn.metrics import roc_curve, auc

# with open(config.newTextPath, "r", encoding="utf-8") as f:
#     lines = f.readlines()

# newText, newVec = utils.lines2data(lines)
# print(newVec.shape)

# scores = np.dot(utils.database['vec'], newVec.T)
# pos = np.argmax(scores>0.8, axis=0)
# for i in range(len(pos)):
#     # fpr,tpr,thresholds=roc_curve(label,scores[pos[i],i])
#     # if scores[pos[i],i]>0.8:
#         print(newText[i])
#         print(utils.database['text'][pos[i]])
#         print(scores[pos[i], i])



import numpy as np

import config
import utils

with open(config.newTextPath, "r", encoding="utf-8") as f:
    lines = f.readlines()

tar = list(range(1, 17))
answers = utils.duplicateCheck(lines, tar, 0.8, 3)

for answer in answers:
    print(answer)

