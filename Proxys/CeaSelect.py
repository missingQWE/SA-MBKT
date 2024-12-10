import os
import copy
import random
import functools

from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torch.utils.tensorboard as tensorboard

from .cea import CEA
from Proxys.utils.util import *
from Proxys.utils import AccuracyMetric, AverageMetric


def CeaSelect(cea, tempPop, baseline, k):
    cea.eval()

    arch_tensor = torch.tensor(tempPop, dtype=torch.float32)
    arch_tensor_shape = arch_tensor.shape
    baseline_tensor = torch.tensor([baseline], dtype=torch.float32)
    baseline_expanded = baseline_tensor.expand(arch_tensor_shape[0], arch_tensor_shape[1])

    arch0 = to_device(arch_tensor)    # 数据迁移到设备
    arch1 = to_device(baseline_expanded)    # 这里要改成基线，不用打乱的了

    outputs = cea(arch0, arch1)
    # select_p, index = torch.topk(outputs, k=k)   # 从预测的概率中选出前k个最大的概率及其对应的索引
    index = torch.nonzero(outputs > 0.5).squeeze()
    # 根据索引选择对应的架构对
    if index.numel() == 0:
        select_pop = []
    else:
        if index.dim() == 0:
            # 如果 index 是一个零维张量，转换为单个元素的列表
            index = [index.item()]
        select_pop = list_select(tempPop, index)

    logger.info(
        ", ".join([
            "CeaSelect \n",
            f"CEAoutputs : {outputs}\n",
            f"baselineFit : {baseline.fitness.values[0]}",
        ])
    )

    return select_pop, outputs

def CEAevaluate(gen, FIToutputs, CEAoutputs, acc):
    accuracy1 = AccuracyMetric()
    loss_avg1 = AverageMetric()
    FIToutputs = torch.tensor(FIToutputs, dtype=torch.float32)
    targets = (FIToutputs > 0).float()

    criterion = nn.BCELoss()
    loss = criterion(CEAoutputs, targets)

    loss_avg1.update(loss)
    accuracy1.update(targets, CEAoutputs)

    logger.info(
        ", ".join([
            "EVAL CEA",
            f"gen={gen:03d}\n",
            f"realFitnesses : {acc}\n",
            f"realTargets : {targets}",
        ])
    )

    logger.info(
        ", ".join([
            f"testAccuracy={accuracy1.compute() * 100:.4f}%",
            f"testLoss={loss_avg1.compute():.4f}",
        ])
    )


