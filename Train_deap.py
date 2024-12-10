import numpy as np
# import joblib
import time
import os
import random
from deap import base, creator, tools
import copy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score
from sklearn import metrics
# from myTools import tools as mytools
from Proxys.GaussProxy import GaussProxy
from myTools.makdir import mkdir
# from Proxys.RBFProxy import RBF
# plt 在Linux使用需要安装另外的软件，暂时先不做图
import matplotlib.pyplot as plt
import csv
# from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as Pool
# plt.style.use('seaborn-whitegrid')

from Proxys.cea import MBSpaceCEA
from Proxys.CeaTrain import CEATrain
from Proxys.CeaSelect import CeaSelect, CEAevaluate
from Proxys.utils.util import *

# 使用AUC作为适应度值，越大越好
creator.create("FitnessMax", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.FitnessMax)

UNMASTEREDID = int(0)
MASTEREDID = int(1)
WRONGID = int(0)
RIGHTID = int(1)

class Trian:
    # winId窗口号，WindowsNum窗口数目，survival上一代窗口存活率，max_generations代，datas当前窗口训练数据，steps答题时间步，KcNums知识点数，length：学生数量
    def __init__(self, winId, WindowsNum, survival, max_generations, datas, steps, KcNums, length, print_str):
        self.print_str = print_str
        self.epochs = 5
        self.id = winId
        # 分的窗口总数
        self.WINNUMS = WindowsNum
        # 上一代窗口个体集成到下一代窗口的比例
        self.survival = 0.3
        self.runs = survival
        self.KcNums = KcNums
        self.AllDatas = datas
        self.AllSteps = steps
        self.length = length
        self.max_generations = max_generations
        self.nowGens = 0
        self.SavePop=None

        # 用于局部搜索的变量， 初始化
        self.PI = np.array([0.8, 0.2])  # 初始状态概率
        self.A = np.random.uniform(0, 1, 4).reshape(2, 2)  # 状态转移矩阵
        self.A = self.A / np.sum(self.A, axis=1).reshape(2, 1)  # 对状态转移概率矩阵 A 进行归一化，以确保每行的概率之和为1
        self.B = np.random.uniform(0, 1, 4).reshape(2, 2)    # 观测概率矩阵
        self.B = self.B / np.sum(self.B, axis=1).reshape(2, 1)

        # 隐状态的个数
        self.N = 2
        # 观测变量的取值个数
        self.M = 2
        # DNA 的类型数 PL， PS PG PF， PW 5或者6
        self.DNATypeNum = 5
        # 种群规模
        self.popSize = 100   # 100

        # 用于代理模型
        # self.ProxyModel = None
        self.ProxyModel = MBSpaceCEA(KcNums * self.DNATypeNum, KcNums * self.DNATypeNum).to(device=device)
        self.proxyTrainData = []
        self.proxyTrainLable = []
        self.trainThreadHoldGens = 1
        self.CproxyModel = None
        self.proxyTrainClabel = []

        # 记录每个窗口评价的个体数
        self.XorList = []
        self.RealXorList = []
        self.XorAllNum = 0
        self.RealXorAllNum = 0
        self.MutList = []
        self.RealMutList = []
        self.MutAllNum = 0
        self.RealMutAllNum = 0
        self.RealSelectEVNum = 0
        # 记录评价每个个体的平均时间（第一代）
        self.avgTime = 0

        # 开始进行演化算法的准备
        self.toolbox = base.Toolbox()  # 创建一个DEAP工具箱对象，这个工具箱用于定义和管理遗传算法的运算符和操作
        # 设置个体的DNA可以是小数,个体的DNA编码是一个随机生成的小数，范围在0到1之间
        self.toolbox.register("attr_random", random.random)  # 在工具箱中注册一个遗传算法的操作，该操作用于生成个体（解的表示）

        # 这是原来的个体生成函数，参考https://blog.csdn.net/m0_37872216/article/details/110652659
        # 基因存储的分别是 PL， PT，s, g, forget, weight(对预测结果的影响权重)
        genes_num = 5 * KcNums
        self.toolbox.register('individual', tools.initRepeat, creator.Individual, self.toolbox.attr_random, n=genes_num)


        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        # 统计数据
        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values)  # 计算适应度的函数
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        # 评价函数
        def evalauate(individual, dirTarget):

            # 排布策略一：
            PL = individual[0: self.KcNums]
            PT = individual[self.KcNums: 2 * self.KcNums]
            s = individual[2 * self.KcNums: 3 * self.KcNums]
            g = individual[3 * self.KcNums: 4 * self.KcNums]
            forget = individual[4 * self.KcNums: 5 * self.KcNums]
            weight = individual[5 * self.KcNums: 6 * self.KcNums]

            # 遍历所有序列，每个学生都判断
            forgetFlag = True

            sum_ = 0
            pred_labels = []
            AllPredictScore = list()
            AllTrueLabel = list()
            for stuIndex in range(0, self.length):      # self.length 表示学生的个数
                self.datas = self.AllDatas[stuIndex]
                self.steps = self.AllSteps[stuIndex]
                print("学生"+str(stuIndex)+"，答题数：", self.datas, "步骤数：", self.steps)

                # 使用前向算法求出alpha（掌握状态矩阵）
                AllTrueLabel.extend(self.datas)
                Alpha = self.forward(self.datas, self.steps, PL, PT, s, g, forget, forgetFlag=True)

                # 开始预测
                T = len(self.datas)
                prob_true, score_pre = np.zeros([T], float), np.zeros([T], float)
                prob_true[0] = 0
                score_pre[0] = 0
                kcTagAppear = [0 for _ in range(self.KcNums)]   # 用于记录知识点的出现次数，初始赋值为0，长度为知识点数
                for t in range(T):
                    kcs_invovle = list(map(int, self.steps[t]))
                    kc_len = len(kcs_invovle)
                    now_probs = []
                    for kc in kcs_invovle:
                        # kc = int(kc)
                        tmpFlag = kcTagAppear[kc] + 1
                        if tmpFlag == 1:    # 知识点 kc 第一次出现在学生的学习
                            update_tmp = np.array([[PL[kc]], [1 - PL[kc]]])     # update_tmp包含两个值：当前知识点的掌握概率和未掌握概率
                        else:
                            T_matrix = np.array([[1 - s[kc], PT[kc]], [s[kc], 1 - PT[kc]]])     # 学习率矩阵（转移）
                            update_tmp = np.dot(T_matrix, Alpha[kc][t - 1:t, :].T)
                        # 计算当前知识点的预测概率
                        update_tmp = update_tmp.flatten()   # 为了计算方便，直接拉平成一维数组
                        # 发射概率, 直接建成一维数组
                        Emit = np.array([1-s[kc], g[kc]])
                        # 多知识点情况下，现在计算出了一个知识点的预测值
                        tmp_prob = (update_tmp * Emit).sum()        # 实际上就是这个公式：Alpha[kc, t, MASTEREDID] * (1-PS[kc]) + (1-Alpha[kc, t, MASTEREDID]) * PG[kc]
                        now_probs.append(tmp_prob)
                    # 把多知识点的预测值取平均，作为这个混合题目的预测值
                    score_pre[t] = np.sum(now_probs) / kc_len
                # 把当前学生的预测值序列添加到预测列表中
                AllPredictScore.extend(list(score_pre))

            # 计算指标，写入文件
            # 计算AUC的新方法
            fpr, tpr, thresholds = metrics.roc_curve(AllTrueLabel, AllPredictScore, pos_label=1)
            auc = metrics.auc(fpr, tpr)

            # 计算准确率
            for i in range(len(AllTrueLabel)):
                pred_labels.append(np.greater_equal(float(AllPredictScore[i]), 0.5).astype(int))

            # 新增评价指标
            C2 = confusion_matrix(AllTrueLabel, pred_labels)
            TP = C2[1][1]
            FP = C2[0][1]
            FN = C2[1][0]
            TN = C2[0][0]

            Precision = TP / (TP + FP)
            Recall = TP / (TP + FN)
            F1 = (2 * Precision * Recall) / (Precision + Recall)
            bb_acc = accuracy_score(AllTrueLabel, pred_labels)

            print("sum: %s, acc: %s, auc: %s, precision: %s, recall: %s, f1: %s " % (
            sum_, bb_acc, auc, Precision, Recall, F1))

            with open(dirTarget + '/' + 'TrainTarget.txt', 'a') as f:
                f.write("sum: %s, acc: %s, auc: %s, precision: %s, recall: %s, f1: %s \n" % (
                    sum_, bb_acc, auc, Precision, Recall, F1))

            # 增加静态惩罚项
            return (auc ,)

        # 注册目标函数
        self.toolbox.register("evaluate", evalauate)
        # 注册交叉运算和变异运算
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=2.0)
        self.toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=60.0, indpb=0.3)
        # 注册tournsize为3 的锦标赛选择:在种群中随机挑选若干子集，并从每个子集中选出适应度最高的个体
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    # 执行一个自定义的前向算法
    def forward(self, obs, steps, PL, PT, PS, PG, PF, weight, forgetFlag):

        '''
        需要先根据假设值，计算出下一时刻（0时刻单独处理）的掌握情况，然后对下一时刻的作答结果进行预测，得到预测值PredScores
        根据下一时刻的真实观测值，更新当前的掌握情况，并继续向后推进
        :param obs:
        :param steps:
        :param PL:
        :param PT:
        :param PS:
        :param PG:
        :param PF:
        :param forgetFlag:
        :return: Alpha PredScores
        '''

        # 0. 初始化
        # 1. 获取真实观测值
        observation = list(map(int, obs))
        T = len(observation)    # 学生观测数据的时间步数(序列长度)

        # kcTagAppear = [0 for _ in range(self.KcNums)] # 用于跟踪知识点出现次数

        #  用于返回所有的预测结果
        PredScores = np.zeros([T], float)   # 记录每个时间步的预测分数
        PredsAfterUpdate = np.zeros([T], float)     # 记录每个时间步更新后的预测分数

        # Aloha 的维度 知识点*时间步数目*2个隐状态
        Alpha = np.random.random([self.KcNums, T, 2])
        Alpha[:, 0, MASTEREDID] = PL    # 时间步0，所有知识点的初始掌握状态都被初始化为PL
        Alpha[:, 0, UNMASTEREDID] = 1.0 - Alpha[:, 0, MASTEREDID]   # 时间步0，所有知识点的初始未掌握状态

        PF = np.array(PF)

        for t in range(T):
            # 预测过程
            kcs_invovle = list(map(int, steps[t]))

            # My predict
            now_probs = []      # 用于存储每个知识点的预测答对概率
            W = []      # 用于存储每个知识点的权重
            PredsAfterUpdate_now_probs = []     # 更新后的预测答对概率
            PredsAfterUpdate_W = []     # 更新后的权重
            # 遍历每个知识点
            for kc in kcs_invovle:

                tmpKcProb = Alpha[kc, t, MASTEREDID] * (1-PS[kc]) + (1-Alpha[kc, t, MASTEREDID]) * PG[kc]   # 答对的概率
                now_probs.append(tmpKcProb)
                W.append(1)

            # 归一化权重
            W = W / np.sum(W)   # 均等策略
            now_probs = np.array(now_probs)     # 比如当前时刻回答了两个知识点，则now_probs中有两个预测概率
            PredScores[t] = np.dot(W, now_probs)    # 按均等策略预测当前答题分数

            if PredScores[t] == np.nan:
                PredScores[t] = 0

            # 依据观测值更新当前的认知状态，同样是只更新相关知识点
            # 先认为没有涉及到知识点，在上一时刻的基础上只有遗忘情况，涉及到的知识点处理后面会重新处理
            if t < T-1:
                if forgetFlag:
                    Alpha[:, t + 1, MASTEREDID] = Alpha[:, t, MASTEREDID] * (1 - PF)
                    Alpha[:, t + 1, UNMASTEREDID] = 1 - Alpha[:, t + 1, MASTEREDID]
                else:
                    Alpha[:, t + 1, MASTEREDID] = Alpha[:, t, MASTEREDID]
                    Alpha[:, t + 1, UNMASTEREDID] = 1 - Alpha[:, t + 1, MASTEREDID]

            ob = observation[t]
            for kc in kcs_invovle:
                # 这里是计算出根据当前作答情况，重新推测当前的掌握概率
                P_Master_fromOB_Now = (ob * Alpha[kc, t, MASTEREDID] * (1-PS[kc]) / (Alpha[kc, t, MASTEREDID] * (1-PS[kc]) + Alpha[kc, t, UNMASTEREDID] * PG[kc])) + \
                    (1 - ob) * Alpha[kc, t, MASTEREDID] * PS[kc] / (Alpha[kc, t, MASTEREDID] * PS[kc] + Alpha[kc, t, UNMASTEREDID] * (1-PG[kc]))
                # 依据重新推测的结果，更新一下当前的掌握情况
                Alpha[:, t, MASTEREDID] = P_Master_fromOB_Now
                Alpha[:, t, UNMASTEREDID] = 1 - Alpha[:, t, MASTEREDID]

                if t < T-1:
                    if forgetFlag:
                        Alpha[kc, t+1, MASTEREDID] = P_Master_fromOB_Now * (1-PF[kc]) + (1 - P_Master_fromOB_Now) * PT[kc]
                        Alpha[kc, t+1, UNMASTEREDID] = 1 - Alpha[kc, t+1, MASTEREDID]
                    else:
                        Alpha[kc, t + 1, MASTEREDID] = P_Master_fromOB_Now + (1 - P_Master_fromOB_Now) * PT[kc]
                        Alpha[kc, t + 1, UNMASTEREDID] = 1 - Alpha[kc, t + 1, MASTEREDID]

                # 以下供检测使用
                PredsAfterUpdate_tmpKcProb = P_Master_fromOB_Now * (1 - PS[kc]) + (1 - P_Master_fromOB_Now) * PG[kc]
                PredsAfterUpdate_now_probs.append(PredsAfterUpdate_tmpKcProb)
                PredsAfterUpdate_W.append(1)

            # 归一化权重
            PredsAfterUpdate_W = PredsAfterUpdate_W / np.sum(PredsAfterUpdate_W)
            PredsAfterUpdate_now_probs = np.array(PredsAfterUpdate_now_probs)
            PredsAfterUpdate[t] = np.dot(PredsAfterUpdate_W, PredsAfterUpdate_now_probs)    # 更新后，对当前知识点答对的概率

        if PredsAfterUpdate[t] == np.nan:
            PredsAfterUpdate[t] = 0

        # Alpha：当前学生更新后的掌握情况，PredScores：当前学生每个时刻预测的分数，PredsAfterUpdate：当前学生每个时刻预测的答对概率
        return Alpha, PredScores, PredsAfterUpdate

    def Myevalauate(self, args):
        '''
        解析个体的DNA，提取出模型的参数（PL、PT、s、g、forget）。
        遍历所有学生的数据，执行前向算法得到预测分数。
        计算性能指标，包括AUC、准确率、精确度、召回率、F1分数等。
        将性能指标写入文件，并返回AUC值作为适应度值。
        '''
        t1 = time.time()

        individual = args[0]
        dirTarget = args[1]

        # 排布策略二：
        PL = individual[0:: self.DNATypeNum]
        PT = individual[1:: self.DNATypeNum]
        s = individual[2::self.DNATypeNum]
        g = individual[3::self.DNATypeNum]
        forget = individual[4::self.DNATypeNum]
        forgetFlag = True
        sum_ = 0
        # 遍历所有序列，每个学生都判断

        pred_labels = []
        # self.length 表示学生的个数
        AllPredictScore = list()
        AllTrueLabel = list()
        # 供测试使用
        PredsAfterUpdate_AllPredictScore = list()
        for stuIndex in range(0, self.length):
            Tdatas = self.AllDatas[stuIndex]
            Tsteps = self.AllSteps[stuIndex]

            # 使用前向算法求出alpha, 通知作出每个时刻的预测值
            if len(Tdatas) == 0:
                continue
            AllTrueLabel.extend(Tdatas)
            # Alpha[t][k]表示在时间步骤t时，学生掌握知识点k的概率
            # Alpha：当前学生更新后的掌握情况，stuPredictScore：当前学生每个时刻预测的分数，stuPredsAfterUpdate_AllPredictScore：当前学生每个时刻预测的答对概率
            Alpha, stuPredictScore, stuPredsAfterUpdate_AllPredictScore = self.forward(Tdatas, Tsteps, PL, PT, s, g, forget, weight=[], forgetFlag=True)
            AllPredictScore.extend(list(stuPredictScore))
            PredsAfterUpdate_AllPredictScore.extend(list(stuPredsAfterUpdate_AllPredictScore))


        # 计算指标，写入文件
        # 计算AUC的新方法
        if np.isnan(AllPredictScore).any():
            auc = 0
        else:
            fpr, tpr, thresholds = metrics.roc_curve(AllTrueLabel, AllPredictScore, pos_label=1)
            auc = metrics.auc(fpr, tpr)

        if np.isnan(PredsAfterUpdate_AllPredictScore).any():
            PredsAfterUpdate_auc = 0
        else:
            PredsAfterUpdate_fpr, PredsAfterUpdate_tpr, PredsAfterUpdate_thresholds = metrics.roc_curve(AllTrueLabel, PredsAfterUpdate_AllPredictScore, pos_label=1)
            PredsAfterUpdate_auc = metrics.auc(PredsAfterUpdate_fpr, PredsAfterUpdate_tpr)

        # 计算准确率
        for i in range(len(AllTrueLabel)):
            TmpPredLabel = np.greater_equal(float(AllPredictScore[i]), 0.5).astype(int)
            pred_labels.append(TmpPredLabel)
            sum_ = sum_ + (1 - abs(AllTrueLabel[i] - TmpPredLabel))     # 表示模型正确分类的样本数

        # 新增评价指标
        C2 = confusion_matrix(AllTrueLabel, pred_labels)
        TP = C2[1][1]
        FP = C2[0][1]
        FN = C2[1][0]
        TN = C2[0][0]

        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = (2 * Precision * Recall) / (Precision + Recall)
        bb_acc = accuracy_score(AllTrueLabel, pred_labels)
        # 计算平方差
        squared_errors = (np.array(AllPredictScore) - np.array(AllTrueLabel)) ** 2
        # 计算均值
        mean_squared_error = np.mean(squared_errors)
        # 计算 RMSE
        rmse = np.sqrt(mean_squared_error)

        print("sum: %s,  acc: %s, auc: %s, precision: %s, recall: %s, f1: %s, rmse: %s " % (
            sum_,  bb_acc, auc, Precision, Recall, F1, rmse))
        print("PredsAfterUpdate_auc: ", PredsAfterUpdate_auc)

        # with open(dirTarget + '/' + 'TrainTarget.txt', 'a') as f:
        with open(dirTarget + '/' + 'tempWithoutLocal.txt', 'a') as f:
            f.write("sum: %s, acc: %s, auc: %s, precision: %s, recall: %s, f1: %s, rmse: %s， time: %s \n" % (
                sum_, bb_acc, auc, Precision, Recall, F1, rmse, time.time() - t1))
            # f.write("评价一次耗时：" + str(time.time() - t1) + 's\n')
        # 增加静态惩罚项
        return (auc,)
        # return (CrossEntropy,)

    def LocalSearchForward(self, observation):
        '''
        计算在给定观察序列情况下，每个时刻处于不同隐含状态的概率
        :param observation: 观测序列。
        :return:
        '''

        T = len(observation)

        # Initialization
        Alpha = np.zeros([2, T], float)
        Alpha[:, 0] = self.PI

        # 0时刻alpha
        Alpha[:, 0] = self.PI * self.B[:, int(float(observation[0]))]   # B是观测概率矩阵
        # 归一化
        Alpha[:, 0] /= Alpha[:, 0].sum()

        for t in range(1, T):
            Alpha[:, t] = np.dot(Alpha[:, t - 1], self.A)   # 计算出在时刻t时各个隐含状态的概率

            Alpha[:, t]   *= self.B[:, int(float(observation[t]))]  # 给定当前观察值后，更新每个隐含状态的概率
            # 归一化
            Alpha[:, t] /= Alpha[:, t].sum()

        return Alpha

    def LocalSearchBackward(self, observation):
        '''
        该函数用于执行后向算法，计算观测序列的隐状态概率分布。
        输入参数 observation 是观测序列。
        函数内部通过迭代计算，得到观测序列每个时刻的隐状态概率分布，并返回结果。
        :param self:
        :param observation:
        :return:
        '''
        T = len(observation)

        Beta = np.zeros([2, T], float)
        Beta[:, T - 1] = 1
        # 归一化
        Beta[:, T - 1] /= Beta[:, T - 1].sum()
        for t in reversed(range(T - 1)):
            tmp = self.A.copy()
            tmp *= self.B[:, int(float(observation[t + 1]))]
            Beta[:, t] = np.dot(tmp, Beta[:, t + 1])
            # 归一化
            Beta[:, t] /= Beta[:, t].sum()

        return Beta

    def LocalSearchBaum_Welch(self, Obs_seq, epochs):
        '''
        该函数用于执行Baum-Welch算法，用于训练隐马尔可夫模型（HMM）的参数，即初始概率向量 PI、状态转移概率矩阵 A、发射概率矩阵 B。
        输入参数 Obs_seq 是多个观测序列的列表。
        函数内部遍历每个观测序列，执行前向算法和后向算法，计算隐状态概率分布，然后利用这些分布来更新HMM模型的参数。
        :param Obs_seq:
        :return:
        '''
        for epoch in range(epochs):
            print("epoch:" + str(epoch))
            exp_si_t0 = np.zeros([2], float)  # [0,0]
            exp_num_from_Si = np.zeros([2], float)  # [0,0]

            exp_num_in_Si = np.zeros([2], float)  # [0,0]

            exp_num_Si_Sj = np.zeros([2 * 2], float).reshape(2, 2)  # [[0,0],[0,0]]

            exp_num_in_Si_Vk = np.zeros([2, 2], float)
            # 训练
            for obs in Obs_seq:
                # print("obs:",obs)
                observation = list(map(int, obs))
                T = len(observation)
                if T == 0:
                    continue
                Alpha = self.LocalSearchForward(observation)
                Beta = self.LocalSearchBackward(observation)

                raw_Gamma = Alpha * Beta
                Gamma = raw_Gamma / raw_Gamma.sum(0)    # 在每个时刻结束于不同的隐含状态的相对概率
                exp_si_t0 += Gamma[:, 0]
                exp_num_from_Si += Gamma[:, :T - 1].sum(1)
                exp_num_in_Si += Gamma.sum(1)

                # Xi is defined as given the model and sequence, the probability of being in state Si at time t,
                # and in state Sj at time t+1
                Xi = np.zeros([T - 1, 2, 2], float)
                for t in range(T - 1):
                    for i in range(2):
                        Xi[t, i, :] = Alpha[i, t] * self.A[i, :]

                        Xi[t, i, :] *= self.B[:, int(float(observation[t + 1]))]
                        Xi[t, i, :] *= Beta[:, t + 1]
                    Xi[t, :, :] /= Xi[t, :, :].sum()

                for t in range(T - 2):
                    exp_num_Si_Sj += Xi[t, :, :]

                # 统计当前序列中的隐状态到观测值的统计状况
                tmp = np.zeros([2, 2], float)
                # 观测值的分别计算
                for each in [0, 1]:
                    which = np.array([each == x for x in observation])  # which中是bool 值序列
                    tmp[:, each] = Gamma.T[which, :].sum(0)
                exp_num_in_Si_Vk += tmp

            # 根据这一epoch 的统计结果，更新各个参数的估计值
            self.PI = exp_si_t0 / exp_si_t0.sum()
            print("self.PI:", self.PI)

            T_hat = np.zeros([2, 2], float).reshape(2, 2)
            for i in range(self.N):
                T_hat[i, :] = exp_num_Si_Sj[i, :] / exp_num_from_Si[i]
                T_hat[i, :] /= T_hat[i, :].sum()
            self.A = T_hat

            E_hat = np.zeros([self.N, self.M], float).reshape(self.N, self.M)
            for i in range(self.N):
                E_hat[i, :] = exp_num_in_Si_Vk[i, :] / exp_num_in_Si[i]
                E_hat[i, :] /= E_hat[i, :].sum()
            self.B = E_hat

    def run(self, sourcePath, winId, dataName, foldName, LastGenPop):

        print_str = self.print_str

        ####################################################设置随机种子，生成当前窗口的初始种群############################################################
        # 判断上一个的种群列表写完没有，LastGenPop存储了上一个窗口的种群
        if winId >= 1 and len(LastGenPop) > 0:
            print("读取上一个窗口的种群！")
            # 生成种群
            random.seed(64)

            pop = LastGenPop[0]
            popGen = self.toolbox.population(n=self.popSize - len(LastGenPop[0]))
            pop = popGen + pop
        else:
            print("上一个窗口没有可读数据，此时的WinID是：", winId, "len(LastGenPop):", len(LastGenPop))
            # print(len(LastGenPop))
            random.seed(64)
            pop = self.toolbox.population(n=self.popSize)
        # 种群中个体数量
        n_pop = len(pop)
        print("种群规模：" + str(n_pop))

        # 每个窗口进来先清空代理模型训练数据
        self.proxyTrainData = []
        self.proxyTrainLable = []

        ##############################################################日志信息准备##########################################################################
        print("Start of evolution")

        # 注册记录，存储日志信息
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + self.stats.fields  # 表头gen nevals avg std min max
        # Evaluate the entire population
        mkdir('./CEAtestResults/')
        mkdir('./CEAtestResults/' + dataName + '/')

        forgetFlag = True
        if not forgetFlag:
            # 如果不允许遗忘，则在每个fold中新建一个文件夹fold/noforget/用来存放各个指标信息
            foldName = foldName + '/noforget'
            # if not os.path.exists(dirPath + '/' + foldName):
            #     mkdir(dirPath + '/' + foldName)
            if not os.path.exists('./CEAtestResults/' + dataName + '/' + foldName):
                mkdir('./CEAtestResults/' + dataName + '/' + foldName)
        else:
            # 考虑遗忘因素，在每个fold中新建一个文件夹fold/forget/用来存放各个指标信息
            # foldName = foldName + '/forget' + str(self.WINNUMS) + 'survival' + str(self.survival)
            foldName = foldName + '/forget' + str(self.WINNUMS) + 'runs' + str(self.runs)
            if not os.path.exists('./CEAtestResults/' + dataName + '/' + foldName):
                mkdir('./CEAtestResults/' + dataName + '/' + foldName)

        mkdir('./CEAtestResults/' + dataName + '/' + foldName + '/winID' + str(winId))
        dirTmp = './CEAtestResults/' + dataName + '/' + foldName + '/winID' + str(winId)
        # with open(dirTmp + '/' + 'TrainTarget.txt', 'a') as f:
        with open(dirTmp + '/' + 'tempWithoutLocal.txt', 'a') as f:
            f.write(print_str)
            f.write("Evaluate Pop:\n")

        with open(dirTmp + '/' + 'templog.csv', 'a', newline='', encoding='utf-8') as f:
            f.write(print_str)
        with open(dirTmp + '/temp.txt', 'a') as f_time:
            f_time.write(print_str)
        ######################################################多线程评估适应度，将其加入代理训练集##########################################################
        # 评价种群个体适应度 使用多进程
        pool = Pool(5)
        pop_iter = [(pop[id], dirTmp) for id in range(n_pop)]
        fitnesses = list(pool.map(self.Myevalauate, pop_iter))  # 以auc的值作为适应度

        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            self.proxyTrainData.append(ind)
            self.proxyTrainLable.append(fit[0])

        # 根据族群适应度，编译出stats记录
        record = self.stats.compile(pop)
        logbook.record(gen=0, nevals=len(pop), **record)  # 第0代
        print("  Evaluated %i individuals" % len(pop))

        ###########################################################开始进化#################################################################################
        gener_num = 0   # 当前代数
        time_start = time.time()
        tempTotalPop = pop  # 当前代的种群
        nowPopNodeThred = 0  # 用作代理模型预测结果是否需要重新实际评估的阈值界限
        # tempBestInd = tools.selBest(pop, 15)[14]


        while (record['max'] - record['avg'] >= 0.005 or record['std'] > 0.0001)  and gener_num < self.max_generations:  # 终止条件
            gener_num = gener_num + 1

            print("-- Generation %i --" % gener_num)
            if (gener_num == 1):
                totalPop = pop
            else:
                totalPop = tempTotalPop
                tempTotalPop = []

            cxpb = (0.8 - ((0.8 - 0.4) / (self.max_generations)) * gener_num)  # 交叉率
            mutpb = (0.8 - ((0.8 - 0.5) / (self.max_generations)) * gener_num)  # 变异率

            # 新增局部最优
            # N_totalPop = len(totalPop)
            N_totalPop = self.popSize
            # 现在采取保留精英的方法，所以先生成一个大于种群数量的子代种群， 再进行淘汰
            offspring = self.toolbox.select(totalPop, N_totalPop)

            # offspring.sort(key=lambda ind: ind.fitness.values, reverse=True)  `# 降序排序

            ###########################################################局部搜索#################################################################################
            # 先进行局部搜索，采用Em算法优化初值
            time_start_localS = time.time()
            self.LocalSearchBaum_Welch(self.AllDatas, self.epochs)
            tim_end_localS = time.time()
            print("局部搜索策略耗时 " + str(tim_end_localS - time_start_localS) + "s")

            # LocalDNA = LocalPL + LocalPT + LocalPS + LocalPG + LocalPF
            # 排布策略二：
            # 使用 creator.Individual（） 把普通列表变成 Individual 类，这是我们自己注册的，继承自List类
            # noisesDNA = [random.uniform(0, 0.01) for _ in range(5 * self.KcNums)]
            LocalDNA = creator.Individual(
                [self.PI[MASTEREDID], self.A[UNMASTEREDID][MASTEREDID], self.A[MASTEREDID][UNMASTEREDID],
                 self.B[MASTEREDID][WRONGID], self.B[UNMASTEREDID][RIGHTID]] * self.KcNums)
            # LocalDNA = creator.Individual([self.PI[MASTEREDID], self.A[UNMASTEREDID][MASTEREDID], self.B[MASTEREDID][WRONGID], self.B[UNMASTEREDID][RIGHTID], self.A[MASTEREDID][UNMASTEREDID]] * self.KcNums )
            print("局部搜索个体：", LocalDNA)

            LocalDNAFit = self.Myevalauate((LocalDNA, dirTmp))

            with open(dirTmp + '/temp.txt', 'a') as f_time:
                f_time.write("当前窗口数： " + str(winId) + ", 进化到第 " + str(gener_num) + " 代耗时：" + str(time.time() - time_start) + "s\n")
                f_time.write("局部搜索个体： " + str(LocalDNA) + "s\n")
                f_time.write("适应度（auc）："+ str(LocalDNAFit) + "s\n")

            LocalSearchPop = [self.toolbox.clone(LocalDNA) for _ in range(10)]
            for tmpId in range(len(LocalSearchPop) - 1):
                noise = [random.uniform(-0.01, 0.01) for _ in range(len(LocalDNA))]
                # 将 noise 加到 individual 的基因上
                LocalSearchPop[tmpId][:] = [min(max(gene + n, 0), 1) for gene, n in zip(LocalSearchPop[tmpId], noise)]
            LocalSearchPop[9].fitness.values = LocalDNAFit
            # # 原来的版本
            # LocalSearchPop = self.toolbox.population(n=10)
            # for tmpId in range(len(LocalSearchPop)):
            #     LocalSearchPop[tmpId] = LocalDNA
            #     LocalSearchPop[tmpId].fitness.values = LocalDNAFit

            time_endLSCreate = time.time()
            print("构造局部搜索个体耗时 " + str(time_endLSCreate - tim_end_localS) + "s")

            offspring = offspring + LocalSearchPop

            ###########################################################交叉变异#################################################################################

            # 后面的交叉变异操作都需要复制原始种群然后进行 交叉变异操作
            ############################# 交叉 #################################
            time_deepCopy_begin = time.time()
            offspring_Xor = [self.toolbox.clone(ind) for ind in offspring]

            time_deepCopy_end = time.time()
            print("XOR深拷贝耗时 " + str(time_deepCopy_end - time_deepCopy_begin) + "s")

            for child1, child2 in zip(offspring_Xor[::2], offspring_Xor[1::2]):

                if random.random() < cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            # 因为之前交叉的个体，删除了他们的适应度值，
            # 所以看一下那些适应度是无效值，说明就是进化的个体，并重新计算即可
            invalid_indXor = [ind for ind in offspring_Xor if not ind.fitness.valid]

            XorLen = len(invalid_indXor)

            XorPop_iter = [(ind, dirTmp) for ind in invalid_indXor]

            time_selectXor_end = time.time()
            print("挑选XOR的个体耗时 " + str(time_selectXor_end - time_deepCopy_end) + "s")

            # 记录各项时间
            with open(dirTmp + '/time.txt', 'a') as f_time:
                f_time.write("进化到第 " + str(gener_num) + "代：\n")
            reEvaluateNumXor = 0
            self.XorAllNum = self.XorAllNum + XorLen
            self.XorList.append(XorLen)

            print("当前窗口数： " + str(winId) + ", 当前代数：" + str(gener_num) + ", XOR交叉产生个体数： " + str(len(invalid_indXor)))
            # 前self.trainThreadHoldGens代正常评估并且记录到代理模型训练集中
            if gener_num <= self.trainThreadHoldGens:
                fitnesses = list(pool.map(self.Myevalauate, XorPop_iter))

                for ind, fit in zip(invalid_indXor, fitnesses):
                    ind.fitness.values = fit
                    self.proxyTrainData.append(ind)
                    self.proxyTrainLable.append(fit[0])
                self.RealXorList.append(XorLen)
                self.RealXorAllNum = self.RealXorAllNum + XorLen

            time_Xor_end = time.time()
            print("XOR 变异耗时 " + str(time_Xor_end - time_selectXor_end) + "s")

            ############################# 变异 ##################################
            offspring_mut = [self.toolbox.clone(ind) for ind in offspring]
            for mutant in offspring_mut:
                # mutate an individual with probability MUTPB
                if random.random() < mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            time_selectMut_end = time.time()
            print("挑选Mut的个体耗时 " + str(time_selectMut_end - time_Xor_end) + "s")
            # 因为之前变异的个体，删除了他们的适应度值，
            # 所以看一下那些适应度是无效值，说明就是进化的个体，并重新计算即可
            invalid_indMut = [ind for ind in offspring_mut if not ind.fitness.valid]
            # dirTrainTarget_mut = [dirTmp] * len(invalid_indMut)
            # fitnesses = map(self.toolbox.evaluate, invalid_indMut, dirTrainTarget_mut)
            MutLen = len(invalid_indMut)
            MutPop_iter = [(ind, dirTmp) for ind in invalid_indMut]
            reEvaluateNumMut = 0
            self.MutList.append(MutLen)
            self.MutAllNum = self.MutAllNum + MutLen
            # 前self.trainThreadHoldGens代正常评估并且记录到代理模型训练集中
            if gener_num <= self.trainThreadHoldGens:
                fitnesses = list(pool.map(self.Myevalauate, MutPop_iter))

                for ind, fit in zip(invalid_indMut, fitnesses):
                    ind.fitness.values = fit
                    self.proxyTrainData.append(ind)
                    self.proxyTrainLable.append(fit[0])
                self.RealMutList.append(MutLen)
                self.RealMutAllNum = self.RealMutAllNum + MutLen

            print("  Evaluated %i individuals(mut)" % len(invalid_indMut))
            time_Mut_end = time.time()
            print("mut 变异耗时 " + str(time_Mut_end - time_selectMut_end ) + "s")

            ###########################################################构建新种群，选择###############################################################################
            if gener_num % 5 == 1:
                # 1. 使用高斯回归模型代理
                trainProxyModelTime = time.time()
                # self.ProxyModel = GaussProxy(self.proxyTrainData, self.proxyTrainLable)

                CEATrain(self.proxyTrainData, self.proxyTrainLable, cea=self.ProxyModel)

            # 原来的方法是交叉变异的后代完全取代上代种群，现在要加入环境选择操作
            offspring = invalid_indMut + invalid_indXor + offspring
            # else:
            #     # 原来的方法是交叉变异的后代完全取代上代种群，现在要加入环境选择操作
            #     offspring = offspring_Xor + offspring_mut + offspring

            # 下面直接进行选择最优个体
            # 原来的select版本
            # pop_selected = []
            valid_offspring = [ind for ind in offspring if ind.fitness.valid]
            invalid_offspring = [ind for ind in offspring if not ind.fitness.valid]

            pop_selected1 = tools.selBest(valid_offspring, N_totalPop, fit_attr='fitness')
            tempBestInd = tools.selBest(pop_selected1, 50)[49]

            temp1 = time.time()
            pop_selected2, CEAoutputs = CeaSelect(self.ProxyModel, invalid_offspring, tempBestInd, 30)
            temp2 = time.time()
            print("代理模型进行预选择耗时： " + str(temp2 - temp1) + "s\n")
            with open(dirTmp + '/Myevalauate.txt', 'a') as f_time:
                f_time.write("代理模型进行一次预选择耗时： " + str(temp2 - temp1) + "s\n")
            # invalid_pop_selected = [ind for ind in pop_selected if not ind.fitness.valid]
            # invalidSelectLen = len(invalid_pop_selected)
            # SelPop_iter = [(ind, dirTmp) for ind in invalid_pop_selected]
            # fitnesses = list(pool.map(self.Myevalauate, SelPop_iter))

            invalidSelectLen = len(pop_selected2)
            if invalidSelectLen > 0:
                SelPop_iter = [(ind, dirTmp) for ind in pop_selected2]
                fitnesses = list(pool.map(self.Myevalauate, SelPop_iter))

                for ind, fit in zip(pop_selected2, fitnesses):
                    ind.fitness.values = fit
                    self.proxyTrainData.append(ind)
                    self.proxyTrainLable.append(fit[0])
            # nowPopNodeThred = max(nowPopNodeThred, tools.selBest(offspring, 1)[0].fitness.values[0])
            self.RealSelectEVNum = self.RealSelectEVNum + invalidSelectLen

            pop_selected = pop_selected1 + pop_selected2
            # tempBestInd = pop_selected[14] # 选择最优个体/平均个体
            time_select_end = time.time()
            print("选择及评估操作耗时 " + str(time_select_end - time_Mut_end) + "s" + "，新产生的个体数： " + str(len(invalid_offspring)) + "重新评估的个体数： " + str(invalidSelectLen))
            with open(dirTmp + '/Myevalauate.txt', 'a') as f_time:
                f_time.write("新产生的个体数： " + str(len(invalid_offspring)) + "，重新评估的个体数： " + str(invalidSelectLen))

            with open(dirTmp + '/data.csv', mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow([gener_num, len(invalid_offspring), invalidSelectLen])
                # # 写入标题行
                # writer.writerow(["Generation", "New Individuals", "Reevaluated Individuals"])
                # # 写入数据行
                # for data in gen_data:
                #     writer.writerow([data["gen_num"], data["invalid_offspring"], data["invalidSelectLen"]])

            # # 测试代理模型效果
            # test_iter = [(ind, dirTmp) for ind in invalid_offspring]
            # test_fitnesses = list(pool.map(self.Myevalauate, test_iter))
            # test_fitnesses_list = [x[0] for x in test_fitnesses]
            # test_outputs = [x - tempBestInd.fitness.values[0] for x in test_fitnesses_list]
            # CEAevaluate(gener_num, test_outputs, CEAoutputs, test_fitnesses_list)

            # 把每一代的个体写成列表并且记录其适应度值，最后进行可视化操作
            # writeData = []
            # writeData2 = []
            # for ind in pop_selected1:
            #     indList = list(ind)
            #     indList.append(ind.fitness.values[0])
            #     writeData.append(indList)
            # with open(dirTmp + '/'+ str(gener_num) + 'ind-1.csv', 'a', newline='', encoding='utf-8') as f:
            #     writer = csv.writer(f)
            #     writer.writerows(writeData)
            # for ind in pop_selected2:
            #     indList2 = list(ind)
            #     indList2.append(ind.fitness.values[0])
            #     writeData2.append(indList2)
            # with open(dirTmp + '/'+ str(gener_num) + 'ind-2.csv', 'a', newline='', encoding='utf-8') as f:
            #     writer = csv.writer(f)
            #     writer.writerows(writeData2)
            # writeData3 = []
            # writeData4 = []
            # for ind in valid_offspring:
            #     indList = list(ind)
            #     indList.append(ind.fitness.values[0])
            #     writeData3.append(indList)
            # with open(dirTmp + '/' + str(gener_num) + 'ind-valid.csv', 'a', newline='', encoding='utf-8') as f:
            #     writer = csv.writer(f)
            #     writer.writerows(writeData3)
            # for ind in invalid_offspring:
            #     indList2 = list(ind)
            #     writeData4.append(indList2)
            # with open(dirTmp + '/' + str(gener_num) + 'ind-invalid.csv', 'a', newline='', encoding='utf-8') as f:
            #     writer = csv.writer(f)
            #     writer.writerows(writeData4)
            time_record_inds = time.time()
            # print("记录每一代的每个个体以及其适应度耗时 " + str(time_record_inds - time_select_end) + "s")


            # tempTotalPop.extend(pop_selected)
            tempTotalPop = pop_selected
            #############################################
            # 每隔10代记录一下DNA的参数
            if gener_num % 10 == 0:
                NowBestInd = tools.selBest(totalPop, 1)[0]
                with open(dirTmp + '/best' + str(gener_num) + '.txt', 'w') as f:
                    f.writelines("best DNA:\n")
                    f.write(str(NowBestInd))
                    f.write(str(NowBestInd.fitness.values) + '\n')

            # with open(dirTmp + '/' + 'TrainTarget.txt', 'a') as f:
            with open(dirTmp + '/' + 'tempWithoutLocal.txt', 'a') as f:
                f.write("-- Generation %i --\n" % gener_num)

            # pop = tools.selBest(offspring, n_pop, fit_attr='fitness')  # 选择精英,保持种群规模
            # Gather all the fitnesses in one list and print the stats


            record = self.stats.compile(tempTotalPop)
            logbook.record(gen=gener_num, nevals=len(tempTotalPop), **record)  # 每一代记录一下

            print("  Min %s" % record['min'])
            print("  Max %s" % record['max'])
            print("  Avg %s" % record['avg'])
            print("  Std %s" % record['std'])
            nowPopNodeThred = max(nowPopNodeThred, record['avg'])
            # avgs.append(mean)
            print("进化到第 " + str(gener_num) + " 代耗时：" + str(time.time() - time_start) + "s")
            with open(dirTmp + '/time.txt', 'a') as f_time:
                f_time.write("当前窗口数： " + str(winId) + ", 进化到第 " + str(gener_num) + " 代耗时：" + str(time.time() - time_start) + "s\n")
                f_time.write("记录每一代的每个个体以及其适应度耗时 " + str(time_record_inds - time_select_end) + "s\n")

            # train_maes.append()

        ###################################################确定最终种群，并留一部分给下个窗口###############################################################
        resultPop = tempTotalPop  # 这么多代后最终的族群
        # 每个窗口的最后一代，只留下50%的个体给下一个窗口
        nowWinPop = tools.selBest(tempTotalPop, int(self.popSize * self.survival), fit_attr='fitness')
        if len(LastGenPop) == 0:
            LastGenPop.append(nowWinPop)
        else:
            LastGenPop[0] = nowWinPop

        print("-- End of (successful) evolution --")
        ###################################################确定最佳个体，把每代种群信息写入日志#############################################################
        best_ind = tools.selBest(resultPop, 1)[0]
        print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

        with open(dirTmp + '/temp.txt', 'a') as f_time:
            f_time.write("当前窗口最佳个体： " + str(best_ind) + "s\n")
            f_time.write("最佳适应度（auc）：" + str(best_ind.fitness.values) + "s\n")

        # 把记录写到文件中
        header = logbook.header
        data = logbook
        # with open(dirTmp + '/' + 'log.csv', 'a', newline='', encoding='utf-8') as f:
        with open(dirTmp + '/' + 'templog.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(data)

        # 记录结果可视化
        # gen = logbook.select('gen')  # 用select方法从logbook中提取迭代次数
        # fit_maxs = logbook.select('max')  # 提取适应度最大值
        # # print("fit_max: ",fit_maxs)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(gen[1:], fit_maxs[1:], 'b-', linewidth=2.0, label='Max Fitness')  # 代数 适应度最大值
        # ax.legend(loc='best')
        # ax.set_xlabel('Generation')
        # ax.set_ylabel('Fitness')
        # plt.show()

        with open(dirTmp + '/' + 'best.txt', 'w') as f:
            f.writelines("best DNA:\n")
            f.write(str(best_ind))
        with open(dirTmp + '/' + 'Myevalauate.txt', 'a') as f:
            f.writelines("avgTime:" + str(self.avgTime) + "\n")
            f.writelines("XorList:" + "\n")
            for t in self.XorList:
                f.write(str(t) + ',')
            f.write('\n')
            f.writelines("RealXorList:" + "\n")
            for t in self.RealXorList:
                f.write(str(t) + ',')
            f.write('\n')
            f.write("共产生Xor个体" + str(self.XorAllNum) + ",其中真实评估的是" + str(self.RealXorAllNum) + '\n')
            f.writelines("MutList:" + "\n")
            for t in self.MutList:
                f.write(str(t) + ',')
            f.write('\n')
            f.writelines("RealMutList:" + "\n")
            for t in self.RealMutList:
                f.write(str(t) + ',')
            f.write('\n')
            f.write("共产生Mut个体" + str(self.MutAllNum) + ",其中真实评估的是" + str(self.RealMutAllNum) + '\n')
            f.write('\n')
            f.write("选择后真实评估的个体数一共是" + str(self.RealMutAllNum) + '\n')


        # plt.plot(gens, avgs)
        # plt.xticks(rotation=90)  # 横坐标每个值旋转90度
        # plt.xlabel('Gens')
        # plt.ylabel('Mean')
        # plt.title('')


        # plt.savefig(dirPath + '/' +  foldName + '/' + 'means.jpg')
        # # plt.legend()
        # plt.show()

        model_param = best_ind
        self.model_param = model_param
        # model_param['PL'] = PL
        # model_param['PT'] = PT
        # model_param['PF'] = PF
        # model_param['s'] = s
        # model_param['g'] = g
        return model_param

    def Test(self, winTestID, dna, datas, seqs, length, dataName, foldName, forgetFlag=True):
        # forgetFlag=True 表示允许遗忘，更符合实际情况，
        # forgetFlag=False 不允许遗忘，更理想化，不太复合实际
        # if self.model_param is not None:
        #     dna = self.model_param
        # 排布策略一：
        # PL = dna[0: self.KcNums]
        # PT = dna[self.KcNums: 2 * self.KcNums]
        # s = dna[2 * self.KcNums: 3 * self.KcNums]
        # g = dna[3 * self.KcNums: 4 * self.KcNums]
        # forget = dna[4 * self.KcNums: 5 * self.KcNums]
        # # weight = dna[5 * self.KcNums: 6 * self.KcNums]

        # 排布策略二:
        PL = dna[0:: self.DNATypeNum]
        PT = dna[1::self.DNATypeNum]
        s = dna[2::self.DNATypeNum]
        g = dna[3::self.DNATypeNum]
        forget = dna[4::self.DNATypeNum]
        # weight = dna[5::self.KcNums]

        Threshold = 0.5
        AllDatas = datas
        AllSteps = seqs
        sum_ = 0
        # 遍历所有序列，每个学生都判断
        # 统计总共的答题步数和预测对的步数
        # 计算评价指标

        pred_labels = []
        test_rmses = []

        AllPredictScore = list()
        AllTrueLabel = list()
        for stuIndex in range(0, length):
            _datas = AllDatas[stuIndex]
            _steps = AllSteps[stuIndex]
            if len(_datas) == 0:
                continue
            AllTrueLabel.extend(_datas)
            Alpha, stuPredictScore, _ = self.forward(_datas, _steps, PL, PT, s, g, forget, weight=[],
                                                     forgetFlag=forgetFlag)
            AllPredictScore.extend(list(stuPredictScore))

            # 开始预测
            # T = len(_datas)
            # prob_true, score_pre = np.zeros([T], float), np.zeros([T], float)
            # prob_true[0] = 0
            # score_pre[0] = 0
            # kcTagAppear = [0 for _ in range(self.KcNums)]
            # for t in range(T):
            #     kcs_invovle = list(map(int, _steps[t]))
            #     # kc_len = len(kcs_invovle)
            #     now_probs = []
            #     W = []
            #     for kc in kcs_invovle:
            #         # kc = int(kc)
            #         kcTagAppear[kc] += 1
            #         if kcTagAppear[kc] == 1:
            #             update_tmp = np.array([[PL[kc]], [1 - PL[kc]]])
            #         else:
            #             T_matrix = np.array([[1 - forget[kc], PT[kc]], [forget[kc], 1 - PT[kc]]])
            #             update_tmp = np.dot(T_matrix, Alpha[kc][t - 1:t, :].T)
            #         # 计算当前知识点的预测概率
            #         update_tmp = update_tmp.flatten()
            #         Emit = np.array([1 - s[kc], g[kc]])
            #         # 多知识点情况下，现在计算出了一个知识点的预测值
            #         tmp_prob = (update_tmp * Emit).sum()
            #         now_probs.append(tmp_prob)
            #         W.append(weight[kc])
            #         # 把多知识点的预测值取平均，作为这个混合题目的预测值
            #     # score_pre[t] = np.nanmean(now_probs)
            #     # MeanTmp = np.nanmean(now_probs)
            #     # score_pre[t] = 1/(1 + np.exp(-MeanTmp))
            #     # 归一化权重
            #     # W = W / np.sum(W)
            #     W = np.exp(W) / sum(np.exp(W))
            #     now_probs = np.array(now_probs)
            #     score_pre[t] = np.dot(W, now_probs)
            # AllPredictScore.extend(list(score_pre))

            # step_index = step_index + 1
        # 统计准确率和总误差
        # 计算指标

        # 计算AUC的方法
        # fpr, tpr, thresholds = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
        # auc = metrics.auc(fpr, tpr)
        # 计算AUC的方法
        fpr, tpr, thresholds = metrics.roc_curve(AllTrueLabel, AllPredictScore, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        # 计算准确率
        for i in range(len(AllTrueLabel)):
            # pred_labels.append(np.greater_equal(float(AllPredictScore[i]), Threshold).astype(int))
            TmpPredLabel = np.greater_equal(float(AllPredictScore[i]), 0.5).astype(int)
            pred_labels.append(TmpPredLabel)
            sum_ = sum_ + (1 - abs(AllTrueLabel[i] - TmpPredLabel))

        # 新增评价指标
        # TN | FP
        # FN | TP
        C2 = confusion_matrix(AllTrueLabel, pred_labels)
        TP = C2[1][1]
        FP = C2[0][1]
        FN = C2[1][0]
        TN = C2[0][0]

        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        F1 = (2 * Precision * Recall) / (Precision + Recall)
        bb_acc = accuracy_score(AllTrueLabel, pred_labels)
        # 计算平方差
        squared_errors = (np.array(AllPredictScore) - np.array(AllTrueLabel)) ** 2
        # 计算均值
        mean_squared_error = np.mean(squared_errors)
        # 计算 RMSE
        rmse = np.sqrt(mean_squared_error)

        APScore = average_precision_score(AllTrueLabel, AllPredictScore)

        print("#####测试开始#####")

        # print("sum: %s, CrossEntropy: %s, acc: %s, auc: %s, precision: %s, recall: %s, f1: %s " % (sum_, CrossEntropy, bb_acc, auc, Precision, Recall, F1))
        print("TESTwinID: %s, sum: %s, acc: %s, auc: %s, precision: %s, recall: %s, f1: %s, rmse: %s, apScore: %s " % (
        winTestID, sum_, bb_acc, auc, Precision, Recall, F1, rmse, APScore))
        # 把测试结果写入文件
        result_dir = './CEAtestResults/' + dataName
        # mkdir(result_dir)
        # mkdir(result_dir + arg[0] + '/')
        # mkdir(result_dir + arg[0] + '/' + str(arg[1]) + '/')
        # mkdir(result_dir + arg[0] + '/' + str(arg[1]) + '/' + arg[2] + '/')
        if not forgetFlag:
            # 如果不允许遗忘，则在每个fold中新建一个文件夹fold/noforget/用来存放各个指标信息
            foldName = foldName + '/noforget'
            if not os.path.exists(result_dir + '/' + foldName):
                mkdir(result_dir + '/' + foldName)
        else:
            # 考虑遗忘因素，在每个fold中新建一个文件夹fold/forget/用来存放各个指标信息
            # foldName = foldName + '/forget' + str(self.WINNUMS) + 'survival' + str(self.survival)
            foldName = foldName + '/forget' + str(self.WINNUMS) + 'runs' + str(self.runs)
            if not os.path.exists(result_dir + '/' + foldName):
                mkdir(result_dir + '/' + foldName)

        mkdir(result_dir + '/' + foldName + '/testWIn' + str(winTestID))
        # with open(result_dir + '/' + foldName + '/testWIn' + str(winTestID) + '/' + 'test_rmse.txt', 'w') as f:
        #     test_rmses = [(str(each) + '\n') for each in test_rmses]
        #     f.writelines(test_rmses)

        with open(result_dir + '/' + foldName + '/testWIn' + str(winTestID) + '/' + 'TestTarget.txt', 'w') as f:
            f.write("Test AUC:" + str(auc) + '\n')
            f.write("Test ACC: " + str(bb_acc) + '\n')
            f.write("Test Precision: " + str(Precision) + '\n')
            f.write("Test Recall: " + str(Recall) + '\n')
            f.write("Test F1: " + str(F1) + '\n')
            f.write("Test RMSE: " + str(rmse) + '\n')
            f.write("Test APScore: " + str(APScore) + '\n')
            # f.write("Test CrossEntropy: " + str(CrossEntropy) + '\n')
            # f.write("Test R2: " + str(r2) + '\n')
            # f.write("Test rmse: " + str(rmse) + '\n')
            # f.write("Test mae: " + str(mae) + '\n')

        # with open(result_dir +  '/' + foldName + '/' + 'test_mae.txt', 'w') as f:
        #     test_maes = [(str(each) + '\n') for each in test_maes]
        #     f.writelines(test_maes)

        # 调试时候可以加上
        with open(result_dir + '/' + foldName + '/testWIn' + str(winTestID) + '/' + 'actu_Label.txt', 'w') as f:
            actual_labels = [(str(each) + '\t') for each in AllTrueLabel]
            f.writelines(actual_labels)

        with open(result_dir + '/' + foldName + '/testWIn' + str(winTestID) + '/' + 'preScores.txt', 'w') as f:
            predScore = [(str(each) + '\t') for each in AllPredictScore]
            f.writelines(predScore)

        # 画PR曲线
        precision, recall, threshold = precision_recall_curve(AllTrueLabel, AllPredictScore, pos_label=1)
        fig = plt.figure()
        plt.plot(precision, recall, label='Logistic')
        # plt.axis([0, 1, 0, 1])
        plt.xlabel('Recall')
        plt.ylabel('Precision')

        plt.legend()
        plt.savefig(result_dir + '/' + foldName + '/testWIn' + str(winTestID) + '/' + 'PR.png')

        print("测试结束！")

        return auc, bb_acc, Precision, Recall, F1, rmse, APScore
