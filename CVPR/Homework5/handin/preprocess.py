import os

datasetpath="Data/256_ObjectCategories"

dirs=os.listdir(datasetpath)
dirs.sort()
with open('label.txt','w',encoding='utf-8') as f:
    for i in dirs:
        f.write(i)
        f.write('\n')

it=0
Matrix = [[] for x in range(257)]                
for d in dirs:
    for _, _, filename in os.walk(os.path.join(datasetpath,d)):
        for i in filename:
            Matrix[it].append(os.path.join(os.path.join(datasetpath,d),i))  
    it = it + 1


with open('dataset-valid.txt','w',encoding='utf-8') as f:
    for i in range(len(Matrix)):
        for j in range(10):
            f.write(os.path.join(Matrix[i][j]))
            f.write(' ')
            f.write(str(i))
            f.write('\n')
with open('dataset-test.txt','w',encoding='utf-8') as f:
    for i in range(len(Matrix)):
        for j in range(10,40):
            f.write(os.path.join(Matrix[i][j]))
            f.write(' ')
            f.write(str(i))
            f.write('\n')
with open('dataset-train.txt','w',encoding='utf-8') as f:
    for i in range(len(Matrix)):
        for j in range(40,len(Matrix[i])):
            f.write(os.path.join(Matrix[i][j]))
            f.write(' ')
            f.write(str(i))
            f.write('\n')

