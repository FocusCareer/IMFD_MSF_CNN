from config import CONFIG
from utils import Accumulator
import torch
from torch import nn
from model import Net
from data_process import train_iter
"""
    points:
        1. net
    边训练边画图: tensorboard
"""
def accuracy_cal(y_hat, y):
    """ (batch_size>1, num_classes>1)"""
    if len(y_hat>1) and y_hat.shape[1]>1:
        y_hat = y_hat.argmax(1)
    cmp = (y_hat.to(y.dtype) == y)
    return cmp.sum().to(torch.float32)

""" 如果想画 loss 曲线或者 accuracy 曲线 """
def train(net, train_iter, loss_func, optimizer, epoches):
    if isinstance(net, torch.nn.Module):
        net.train()
    
    for epoch in range(epoches):
        metric = Accumulator(3)
        for x, y in train_iter:
            pred = net(x)
            # print(pred.shape, y.shape)
            loss = loss_func(pred, y)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            metric.add(float(loss.sum()), accuracy_cal(pred, y), y.numel())
        
        loss_e = metric[0] / metric[2]
        accuracy = metric[1] / metric[2]
        print(f"epoch {epoch+1}, loss: {loss_e} acc:{accuracy}")

""" 应用于测试集 """
def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    """ 提高计算效率 """
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy_cal(net(X), y), y.numel())
    return metric[0] / metric[1]

if __name__ == "__main__":
    net = Net(in_channels=1, negative_slope=0.1)
    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=CONFIG["lr"])
    train(net, train_iter, loss_f, optimizer, CONFIG["epoches"])
