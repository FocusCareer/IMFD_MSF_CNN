import torch
from torch import nn 
from sklearn.decomposition import PCA    #在sklearn中调用PCA机器学习算法
                  #定义所需要分析主成分的个数n
""" 
    x_train: (500, 8)
    pca(x_train): (500, 3)
"""
def pca_fit(x_train, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(x_train)

def pca_transform_dim(x, pca):
    return pca.transform(x)

""" (8, N, N) -> ()"""
def random_sampling(dataset):
    pass

shape = (1000, 3, 28, 28)
num_classes = [4]
x_train = torch.rand(shape)
y_train = torch.randint(0, 3, (1000,))

shape_2 = (100, 3, 28, 28)
x_test = torch.rand(shape_2)
y_test = torch.randint(0, 3, (100,))

from config import CONFIG
# from torch.utils.data import DataLoader, TensorDataset

# train_dataset = TensorDataset(x_train, y_train)
# train_iter = DataLoader(train_dataset, shuffle=True, batch_size=CONFIG["batch_size"])

# test_dataset = TensorDataset(x_test, y_test)
# data_iter = DataLoader(test_dataset, shuffle=True, batch_size=CONFIG["batch_size"])

import torch
from torch import nn
from torch.nn import init
import numpy as np
import sys
import torchvision
import torchvision.transforms as transforms
import time
mnist_train = torchvision.datasets.FashionMNIST(root='/home/aguang/gqh/data/', train=True, download=False, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='/home/aguang/gqh/data/', train=False, download=False, transform=transforms.ToTensor())

def load_data_fashion_mnist(mnist_train, mnist_test, batch_size):
    if sys.platform.startswith('win'):
        num_workers = 0
    else:
        num_workers = 4
    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter

batch_size = CONFIG["batch_size"]
train_iter, test_iter = load_data_fashion_mnist(mnist_train, mnist_test, batch_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
