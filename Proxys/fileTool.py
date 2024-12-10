import json

from torch.utils.data import TensorDataset

from cea import CEA, MBSpaceCEA
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
from utils.util import device
from CeaTrain import train, evaluate, CEATrain
from utils.util import *
# from GaussProxy import GaussProxy

def read_traindata(file_path):
    """读取traindata文件并将其内容转换为二维数组"""
    with open(file_path, 'r') as file:
        data = json.loads(file.read())
    return data

def read_trainlabels(file_path):
    """读取trainlabel文件并将其内容转换为一维数组"""
    with open(file_path, 'r') as file:
        labels = json.loads(file.read())
    return labels

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# 文件路径
traindata_file = 'C:/Users/14346/Desktop/traindata.txt'
trainlabel_file = 'C:/Users/14346/Desktop/trainlabel.txt'

# 读取训练数据和训练标签
train_data = read_traindata(traindata_file)
train_labels = read_trainlabels(trainlabel_file)


outputs = torch.tensor([0.3, 0.6, 0.7])
index = torch.nonzero(outputs > 0.5).squeeze()

# 根据索引选择对应的架构对
if index.numel() == 0:
    select_pop = []
else:
    if index.dim() == 0:
        # 如果 index 是一个零维张量，转换为单个元素的列表
        index = [index.item()]

    select_pop = list_select(train_data, index)



