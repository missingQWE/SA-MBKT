import torch
import torch.nn as nn

# from .gcn import GCN

# from core.dataset.tensorize import nasbench_tensor2arch
# from core.utils import _sign
# from core.config import args


class CEA(nn.Module):
    def __init__(self, n_nodes, n_ops, n_layers=2, ratio=2, embedding_dim=128):
        super(CEA, self).__init__()
        self.n_nodes = n_nodes
        self.n_ops = n_ops

        self.embedding = nn.Embedding(self.n_ops+3, embedding_dim=embedding_dim)

        #todo  MLP初始化
        self.MLP = MLPo(125,100,85)

        self.fc = nn.Linear(embedding_dim*ratio, 1, bias=True)

    def forward(self, arch0, arch1):

        feature = torch.cat([self.extract_features(arch0), self.extract_features(arch1)], dim=1)

        score = self.fc(feature).view(-1)

        probility = torch.sigmoid(score)

        return probility

    def extract_features(self, arch):
        emd = self.MLP(arch)
        return emd


class MLPo(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation=nn.ReLU):
        """
        初始化MLP类

        参数:
        input_size (int): 输入层的维度
        hidden_sizes (list of int): 隐藏层的维度列表
        output_size (int): 输出层的维度
        activation (class): 激活函数类，默认为nn.ReLU
        """
        super(MLPo, self).__init__()

        # 定义一个列表来存储所有的层
        layers = []

        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation())

        # 隐藏层之间的连接
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            layers.append(activation())

        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # 将所有层放入一个有序字典中
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播函数

        参数:
        x (Tensor): 输入张量

        返回:
        Tensor: 输出张量
        """
        return self.model(x)

def linear_bn_relu(in_features, out_features, relu=True):
    layers = [nn.Linear(in_features, out_features)]
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return layers

class MLP(nn.Sequential):
    def __init__(self, in_features, hidden_features, out_features, n_layers):
        assert n_layers > 1
        layers = []
        for i in range(n_layers):
            in_f = in_features if i == 0 else hidden_features
            out_f = out_features if i == n_layers-1 else hidden_features
            relu = i != n_layers-1
            layers += linear_bn_relu(in_f, out_f, relu)
        super(MLP, self).__init__(*layers)


class MBSpaceCEA(nn.Module):
    def __init__(self, in_features, hidden_features, n_layers=3):
        super(MBSpaceCEA, self).__init__()

        self.linear_extractor = MLP(in_features, hidden_features, hidden_features, n_layers)

        self.fc = nn.Linear(hidden_features*2, 1, bias=True)

    def forward(self, arch0, arch1):
        if isinstance(arch0, list):
            arch0 = arch0[0]
        if isinstance(arch1, list):
            arch1 = arch1[0]
        # import ipdb; ipdb.set_trace()
        feature = torch.cat([self.linear_extractor(arch0), self.linear_extractor(arch1)], dim=1)
        score = self.fc(feature).view(-1)
        probility = torch.sigmoid(score)

        return probility
