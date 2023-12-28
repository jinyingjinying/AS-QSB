# -*- coding: utf-8 -*-
"""
Created on Wed July  12 13:15:10 2023

@author: yjin
"""


import numpy as np
import pandas as pd
import os
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pandas.core.frame import DataFrame

# Change the current working directory to the specified path
os.chdir('d://document//AS-QSB//machine_learning')

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = nn.Linear(20, 16)
        self.l2 = nn.Linear(16, 8)
        self.l3 = nn.Linear(8, 4)
        self.l4 = nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)


def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        data = torch.tensor(data, dtype=torch.float32)
        output = model(data)
        loss = criterion(output, target.long())
        train_loss += loss.data.item()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    print('Train Epoch: {}, Average loss: {:.4f}'.format(epoch, train_loss))


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for data, target in test_loader:

        data, target = Variable(data, volatile=True),Variable(target)
        data = torch.tensor(data, dtype=torch.float32)
        output = model(data)
        test_loss += criterion(output, target.long()).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        tp += ((pred == 1) & (target.data.view_as(pred) == 1)).cpu().sum()
        tn += ((pred == 0) & (target.data.view_as(pred) == 0)).cpu().sum()
        fn += ((pred == 0) & (target.data.view_as(pred) == 1)).cpu().sum()
        fp += ((pred == 1) & (target.data.view_as(pred) == 0)).cpu().sum()
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)
    accuracy = 1. * (tp + tn) / (tp + tn + fp + fn)
    precision = 1. * tp / (tp + fp)
    sensitive = 1. * tp / (tp + fn)
    f1 = 2 * precision * sensitive / (precision + sensitive)

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}, '
          'Precision: {:.4f}, Sensitive: {:.4f}, F1: {:.4f}\n'.format(
            test_loss, accuracy, precision, sensitive, f1)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--epochs", type=int, default=30)
    parser.add_argument("-bs", "--batch_size", type=int, default=128)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-dp", "--data_path", type=str, default="data/train/merge.csv")
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    data_path = args.data_path

    torch.manual_seed(42)  
    start = time.time()

    data = pd.read_csv(data_path)
    data = np.array(data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)

    train_set, test_set = train_test_split(data, test_size=0.2, random_state=36)

    X_train = train_set[:, 0:-1]
    X_train_label = train_set[:, [-1]]  
    X_train_label = X_train_label.reshape(train_set.shape[0], )
    X_train = torch.from_numpy(X_train.astype(float))
    X_train_label = torch.from_numpy(X_train_label.astype(float))
    train_dataset = TensorDataset(X_train, X_train_label)

    X_test = test_set[:, 0:-1]  
    X_test_label = test_set[:, [-1]]  
    X_test_label = X_test_label.reshape(test_set.shape[0], )
    X_test = torch.from_numpy(X_test.astype(float))
    X_test_label = torch.from_numpy(X_test_label.astype(float))
    test_dataset = TensorDataset(X_test, X_test_label)

    # training setting
    batch_size = batch_size

    # load data
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    model = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), 
                          lr=learning_rate, 
                          momentum=0.5)

    for epoch in range(1, epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer)
        test(model, test_loader, criterion)

    print('train done!')
    
    
    # load data
    data_pre = pd.read_csv('data/predict/merge.csv', sep = ',')
    data = np.array(data_pre)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data_pre)
    print('load data and scaler done!')    
    
    
    # predict
    name_list = []
    data = torch.tensor(data, dtype=torch.float32)
    output = model(data)
    to_list = output.data.max(1, keepdim=True)[1]
    print('predict done!')
    
    for i in to_list:
        name_list.append(i)
    name = DataFrame(name_list)
    print('dataframe done!')
    
    # results output
    name.to_csv('output/dnn_results2.csv',encoding = 'utf-8')
    print('output done!')
    
    end = time.time()
    elapsed_time = end - start 
    minutes, seconds = divmod(elapsed_time, 60) 
    print(f"Elapsed time: {int(minutes)} minutes {int(seconds)} seconds")
    
    
if __name__ == '__main__':
    main()
