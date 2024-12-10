import random
import functools

import torch
import torch.nn.functional as F

from Proxys.utils import AccuracyMetric, AverageMetric
from Proxys.utils.util import *
from Proxys.cea import CEA, MBSpaceCEA
import numpy as np

from torch.utils.data import TensorDataset

import torch.optim as optim

import torch.nn as nn


def CEATrain(train_data, train_labels, cea):
    arch_tensor = torch.tensor(train_data, dtype=torch.float32)
    acc_tensor = torch.tensor(train_labels, dtype=torch.float32)

    # 创建 TensorDataset
    dataset = TensorDataset(arch_tensor, acc_tensor)
    # trainset = CachedSubset(dataset, list(range(50)))  # 50
    train_loader = data.DataLoader(dataset, batch_size=50, shuffle=True, num_workers=0)

    # 节点数5，可操作数3，层数4
    # cea = CEA(n_nodes=5, n_ops=3, ratio=2, n_layers=4, embedding_dim=128).to(device=device)
    # cea = MBSpaceCEA(Embedding_DIM, Embedding_DIM).to(device=device)

    cea_optimizer = optim.Adam(cea.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=5e-4)

    criterion = nn.BCELoss()
    best_KTau = -1.0
    # CEA训练
    for epoch in range(1, 101):
        accuracy, loss = train(epoch=epoch, labeled_loader=train_loader, pseudo_set=None,
                               pseudo_ratio=1.0,
                               cea=cea, criterion=criterion, optimizer=cea_optimizer)
        if accuracy > 0.99 or loss < 0.04:
            print(f"Early stopping at epoch {epoch} with accuracy {accuracy:.4f} and loss {loss:.4f}")
            break  # 如果条件满足，则提前退出循环
    return accuracy, loss

def train(epoch, labeled_loader, pseudo_set, pseudo_ratio, cea, criterion, optimizer, report_freq=10):
    cea.train()
    accuracy = AccuracyMetric()
    loss_avg = AverageMetric()

    for iter_, (arch, acc) in enumerate(labeled_loader, start=1):
        arch0, acc0 = to_device(arch, acc)    # 数据迁移到设备
        arch1, acc1 = shuffle(arch0, acc0)    # 打乱
        targets = (acc0 > acc1).float()
        if pseudo_set is not None and pseudo_ratio != 0:
            batch_size = int(acc0.shape[0] * pseudo_ratio)
            pseudo_set_size = len(pseudo_set[0][0])
            index = random.sample(list(range(pseudo_set_size)), batch_size)
            un_arch0, un_arch1, pseudo_labels = pseudo_set
            un_arch0 = list_select(un_arch0, index)
            un_arch1 = list_select(un_arch1, index)
            pseudo_labels = pseudo_labels[index]
            # import ipdb; ipdb.set_trace()
            arch0 = concat(arch0, un_arch0)
            arch1 = concat(arch1, un_arch1)
            targets = torch.cat([targets, pseudo_labels], dim=0)

        optimizer.zero_grad()

        outputs = cea(arch0, arch1)

        loss = criterion(outputs, targets)

        loss_avg.update(loss)
        accuracy.update(targets, outputs)

        loss.backward()
        optimizer.step()

    logger.info(
        ", ".join([
            "TRAIN Complete",
            f"epoch={epoch:03d}",
            f"accuracy={accuracy.compute()*100:.4f}%",
            f"loss={loss_avg.compute():.4f}",
        ])
    )
    return accuracy.compute(), loss_avg.compute()

#
# def evaluate(epoch, loader, cea):
#     cea.eval()
#
#     KTau = AverageMetric()
#
#     for iter_, (*arch, acc, _) in enumerate(loader, start=1):
#         *arch, acc = to_device(*arch, acc)
#
#         KTau_ = compute_kendall_tau_AR(cea, arch, acc)
#
#         KTau.update(KTau_)
#
#         logger2.info(
#             ", ".join([
#                 "EVAL Complete" if iter_ == len(loader) else "EVAL",
#                 f"epoch={epoch:03d}",
#                 f"iter={iter_:03d}",
#                 f"KTau={KTau_:.4f}({KTau.compute():.4f})",
#             ])
#         )
#     return KTau.compute()

def evaluate(epoch, loader, criterion, cea, Proxy):
    cea.eval()
    accuracy1 = AccuracyMetric()
    loss_avg1 = AverageMetric()
    accuracy2 = AccuracyMetric()
    loss_avg2 = AverageMetric()

    for iter_, (arch, acc) in enumerate(loader, start=1):
        arch0, acc0 = to_device(arch, acc)  # 数据迁移到设备
        arch1, acc1 = shuffle(arch0, acc0)  # 打乱
        targets = (acc0 > acc1).float()

        outputs = cea(arch0, arch1)

        loss = criterion(outputs, targets)

        loss_avg1.update(loss)
        accuracy1.update(targets, outputs)

        logger.info(
            ", ".join([
                "EVAL Complete",
                f"epoch={epoch:03d}",
                f"testAccuracy={accuracy1.compute() * 100:.4f}%",
                f"testLoss={loss_avg1.compute():.4f}",
            ])
        )

        arch2 = arch0.cpu()
        arch3 = arch1.cpu()
        outputs1 = torch.tensor(Proxy.predict(np.array(arch2)))
        outputs2 = torch.tensor(Proxy.predict(np.array(arch3)))

        outputs_G = (outputs1 > outputs2).float()

        loss_G = criterion(outputs_G, targets.cpu())

        loss_avg2.update(loss_G)
        accuracy2.update(targets.cpu(), outputs_G)
        logger.info(
            ", ".join([
                "GaussProxy EVAL Complete",
                f"epoch={epoch:03d}",
                f"testAccuracy={accuracy2.compute() * 100:.4f}%",
                f"testLoss={loss_avg2.compute():.4f}",
            ])
        )
        logger.info(
            ", ".join([
                f"LABEL ACC0: {acc0}",
            ])
        )
        logger.info(
            ", ".join([
                f"LABEL ACC1: {acc1}",
            ])
        )
        logger.info(
            ", ".join([
                f"LABEL TARGET: {targets}",
            ])
        )
        logger.info(
            ", ".join([
                f"CEA outputs: {outputs}",
            ])
        )
        logger.info(
            ", ".join([
                f"GaussProxy outputs1: {outputs1}",
            ])
        )
        logger.info(
            ", ".join([
                f"GaussProxy outputs2: {outputs2}",
            ])
        )
    return accuracy1.compute(), loss_avg1.compute()