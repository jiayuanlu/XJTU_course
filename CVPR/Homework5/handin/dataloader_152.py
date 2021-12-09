import torch
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch, torchvision
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from torchsummary import summary
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from tqdm import tqdm
root='./'
dataset='Data/256_ObjectCategories/'
num_classes=257
 
def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

transform = transforms.Compose([
    transforms.Scale(257),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std),
    ])

train_data = MyDataset(txt=root+'dataset-train.txt', transform=transform)
test_data = MyDataset(txt=root+'dataset-test.txt', transform=transform)
val_data=MyDataset(txt=root+'dataset-valid.txt', transform=transform)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)
val_loader=DataLoader(dataset=val_data,batch_size=64)


resnet152 = models.resnet152(pretrained=True)
for param in resnet152.parameters():
    param.requires_grad = False
fc_inputs = resnet152.fc.in_features
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
resnet152.fc = nn.Sequential(
    nn.Linear(fc_inputs, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 257), 
    nn.LogSoftmax(dim=1)).to(device) 
resnet152 = resnet152.to('cuda:0')
loss_func = nn.NLLLoss()
optimizer = optim.Adam(resnet152.parameters())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def train_and_validate(model, loss_criterion, optimizer, epochs=25):
    history = []
 
    for epoch in range(epochs):
        epoch_start = time.time()
        print("Epoch: {}/{}".format(epoch+1, epochs))

        model.train()

        train_loss = 0.0
        train_acc = 0.0
        
        valid_loss = 0.0
        valid_acc = 0.0
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            train_acc += acc.item() * inputs.size(0)

        with torch.no_grad():
            model.eval()
            for inputs, labels in tqdm(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = loss_criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, predictions = torch.max(outputs.data, 1)
                correct_counts = predictions.eq(labels.data.view_as(predictions))
                acc = torch.mean(correct_counts.type(torch.FloatTensor))
                valid_acc += acc.item() * inputs.size(0)
            
        avg_train_loss = train_loss/len(train_data)
        avg_train_acc = train_acc/len(train_data)

        avg_valid_loss = valid_loss/len(val_data)
        avg_valid_acc = valid_acc/len(val_data)
 
        history.append([avg_train_loss, avg_valid_loss, avg_train_acc, avg_valid_acc])                
        epoch_end = time.time()
        print("Epoch : {:03d}, Training: Loss: {:.4f}, Accuracy: {:.4f}%, \n\t\tValidation : Loss : {:.4f}, Accuracy: {:.4f}%, Time: {:.4f}s".format(epoch, avg_train_loss, avg_train_acc*100, avg_valid_loss, avg_valid_acc*100, epoch_end-epoch_start))
        torch.save(model, dataset+'_model152_'+str(epoch)+'.pt')            
    return model, history

num_epochs = 30
trained_model, history = train_and_validate(resnet152, loss_func, optimizer, num_epochs)
torch.save(history, dataset+'_history152.pt')

history = np.array(history)
plt.plot(history[:,0:2])
plt.legend(['Train Loss', 'Valid Loss'])
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.ylim(0,1)
plt.savefig(dataset+'_loss_curve152.png')
plt.show()
plt.plot(history[:,2:4])
plt.legend(['Train Accuracy', 'Valid Accuracy'])
plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.ylim(0,1)
plt.savefig(dataset+'_accuracy_curve152.png')
plt.show()


def computeTestSetAccuracy(model, loss_criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_acc = 0.0
    test_loss = 0.0
    history=[]
    with torch.no_grad():
        model.eval()
        for j, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = loss_criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            _, predictions = torch.max(outputs.data, 1)
            correct_counts = predictions.eq(labels.data.view_as(predictions))
            acc = torch.mean(correct_counts.type(torch.FloatTensor))
            test_acc += acc.item() * inputs.size(0)
 
            print("Test Batch number: {:03d}, Test: Loss: {:.4f}, Accuracy: {:.4f}".format(j, loss.item(), acc.item()))

    avg_test_loss = test_loss/len(test_data)
    avg_test_acc = test_acc/len(test_data)
    history.append([avg_test_loss, avg_test_acc])
    print("Test accuracy : " + str(avg_test_acc))
    print("Test loss : " + str(avg_test_loss))
    return history

model=torch.load(dataset+'_model152_'+str(29)+'.pt')
history_test = computeTestSetAccuracy(model, loss_func)
torch.save(history_test, dataset+'_history_test152.pt')


# history_test = np.array(history_test)
# plt.plot(history_test[:,0:2])
# plt.legend(['Test Loss'])
# plt.xlabel('Epoch Number')
# plt.ylabel('Loss')
# plt.ylim(0,1)
# plt.savefig(dataset+'_loss_curve_test152.png')
# plt.plot(history_test[:,1])
# plt.legend(['Test Accuracy'])
# plt.xlabel('Epoch Number')
# plt.ylabel('Accuracy')
# plt.ylim(0,1)
# plt.savefig(dataset+'_accuracy_curve_test152.png')
# plt.show()
