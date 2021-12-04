from __future__ import print_function
import argparse
import torch
# torch.cuda.set_device(0)
import nni
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
criterion = nn.CrossEntropyLoss()
import numpy as np
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from nni.utils import merge_parameter
writer = SummaryWriter('logs/mnist_experiment_1')
logger = logging.getLogger('mnist_AutoML')
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # prelu=nn.PReLU(num_parameters=1)
        # self.dropout1 = nn.Dropout2d(0.25)
        self.in_hid_1= nn.Linear(784, 512) 
        self.hid1=nn.LeakyReLU() 
        self.in_hid_2= nn.Linear(512, 256)  
        self.hid2=nn.LeakyReLU()
        self.in_hid_3= nn.Linear(256, 128)  
        self.hid3=nn.LeakyReLU()
        self.hid_out=nn.Linear(128,10)

    def forward(self, data):
        x = data.view(-1, 784)
        output=self.in_hid_1(x)
        # output=self.dropout1(output)
        output=self.hid1(output)
        output=self.in_hid_2(output)   
        output=self.hid2(output)
        output=self.in_hid_3(output)   
        output=self.hid3(output)
        output=self.hid_out(output)
        output = F.log_softmax(output, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)  
        correct += pred.eq(target.view_as(pred)).sum().item()
        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if batch_idx != 0:
                global_step = (epoch - 1) * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train', running_loss / (args['batch_size'] * args['log_interval']), global_step)
                writer.add_scalar('Accuracy/train', 100. * correct / (args['batch_size'] * args['log_interval']), global_step)
            running_loss = 0.0
            correct = 0.0

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() 
            pred = output.argmax(dim=1, keepdim=True) 
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def profile(model, device, train_loader):
    dataiter = iter(train_loader)
    data, target = dataiter.next()
    data, target = data.to(device), target.to(device)
    with torch.autograd.profiler.profile(use_cuda=False) as prof:
        model(data[0].reshape(1,1,28,28))
    print(prof)

def main():
    torch.backends.cudnn.enabled = False ###
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    # use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)

    # device = torch.device("cuda" if use_cuda else "cpu")
    # kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('data', train=False, transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()

    # grid = torchvision.utils.make_grid(images)
    # writer.add_image('images', grid, 0)

    # model = Net().to(device)
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    # images=images.to(device)
    # writer.add_graph(model, images)
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    # print("Start profiling...")
    # profile(model, device, train_loader)
    # print("Finished profiling.")
    # for epoch in range(1, args.epochs + 1):
    #     train(args, model, device, train_loader, optimizer, epoch)
    #     test_acc=test(model, device, test_loader)
    #     scheduler.step()
    #     # report intermediate result
    #     nni.report_intermediate_result(test_acc)
    #     logger.debug('test accuracy %g', test_acc)
    #     logger.debug('Pipe send intermediate result done.')
 
    # # report final result
    # nni.report_final_result(test_acc)


    # if args.save_model:
    #     print("Our model: \n\n", model, '\n')
    #     print("The state dict keys: \n\n", model.state_dict().keys())
    #     torch.save(model.state_dict(), "mnist.pt")
    #     state_dict = torch.load('mnist.pt')
    #     print(state_dict.keys())
    # writer.close()

    return args

def NNI(args):
    # use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()
    # torch.manual_seed(args.seed)
    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args['batch_size'], shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data', train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args['test_batch_size'], shuffle=True, **kwargs)
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    grid = torchvision.utils.make_grid(images)
    writer.add_image('images', grid, 0)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args['lr'])
    images=images.to(device)
    writer.add_graph(model, images)
    scheduler = StepLR(optimizer, step_size=1, gamma=args['gamma'])#等间隔调整学习率 StepLR， 将学习率调整为 lr*gamma
    print("Start profiling...")
    profile(model, device, train_loader)
    print("Finished profiling.")
    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc=test(model, device, test_loader)
        scheduler.step()
        nni.report_intermediate_result(test_acc)
        logger.debug('test accuracy %g', test_acc)
        logger.debug('Pipe send intermediate result done.')
    nni.report_final_result(test_acc)


    if args['save_model']:
        print("Our model: \n\n", model, '\n')
        print("The state dict keys: \n\n", model.state_dict().keys())
        torch.save(model.state_dict(), "mnist.pt")
        state_dict = torch.load('mnist.pt')
        print(state_dict.keys())
    writer.close()


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(main(), tuner_params))
    print(params)
    NNI(params)
