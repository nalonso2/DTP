import torch
from torch import nn

def to_one_hot(y_onehot, y):
    y = y.view(y_onehot.size(0), -1)
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)


def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    total = labels.size(0)
    correct = (predicted == labels).sum().item()

    return correct / total

def compute_num_correct(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    return correct