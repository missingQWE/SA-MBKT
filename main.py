import time
from myTools import tools as mytools
from Train_deap import Trian as Train_deap
import math
import csv
from myTools.makdir import mkdir
import os

def Read_X(folder):
    # folder表示文件存放地址
    Datas = list()
    Steps = list()
    with open(folder, 'r', encoding='utf-8') as f:
        old_lines = f.readlines()
    lines_steps = []
    lines_ans = []
    for idx in range(len(old_lines)):
        if idx % 3 == 1:
            lines_steps.append(old_lines[idx])
        if idx % 3 == 2:
            lines_ans.append(old_lines[idx])
    # 读取结果数据和步骤数据
    del old_lines
    # 1. 结果数据
    for line in lines_ans:
        line_list = line.strip().split('\t')
        new_line_list = list()
        for each in line_list:
            each = int(float(each))
            # each = round(each, 3)
            # if each >= 0:
            #     new_line_list.append(each)
            new_line_list.append(each)
        Datas.append(new_line_list)

    # 2. 步骤数据
    for line in lines_steps:
        line_list = line.strip().split('\t')
        new_line_list = list()
        for each in line_list:
            CurrKcs = each.split('_')
            new_line_list.append(CurrKcs)
        Steps.append(new_line_list)
    del lines_steps
    del  lines_ans
    return Datas, Steps


def  single_run(args):
    '''
    训练过程
    :param args: 包含训练数据集的路径，包括该数据集的信息Q矩阵
    :return: 各项指标acc mae rmse
    '''
    # 1. 拿到数据
    sourcePath = args[0] # '../DataSets'
    dataName = args[1]     # algebra05
    foldName = args[2]    # fold0~ fold4
    kc_nums = args[3]
    max_generations = args[4]

    # 分的窗口数
    WindowsNum = args[5]
    # 上一代窗口存活比例
    survival = args[6]  # 修改后其实是runs轮次
    print_str = args[7]
    # print(sourcePath + '/' + dataName + '/MultiBKT_data/' + foldName + '/training.txt', "窗口数：", WindowsNum, "上一代窗口存活比例：", survival)
    print(sourcePath + '/' + dataName + '/MultiBKT_data/' + foldName + '/training.txt', "窗口数：", WindowsNum, "第：", survival, "轮")

    #######################################################读取数据######################################################################
    All_train_datas, All_train_steps = Read_X(sourcePath + '/' + dataName + '/MultiBKT_data/' + foldName + '/training.txt')
    All_test_datas, All_test_steps = Read_X(sourcePath + '/' + dataName + '/MultiBKT_data/' + foldName + '/testing.txt')
    print("读取数据成功")

    #######################################################按窗口处理数据################################################################
    paramList = list()
    Train = None
    LastGenPop = list()
    windowSize = math.ceil(len(All_train_datas) / WindowsNum)

    for winId in range(1):

        test_datas = list()
        test_steps = list()

        # 获取当前窗口的训练数据
        if winId * windowSize >= len(All_train_datas):
            break
        train_datas = All_train_datas[winId * windowSize : (winId+1) * windowSize]
        train_steps = All_train_steps[winId * windowSize : (winId+1) * windowSize]
        print("训练窗口号:" + str(winId) + ", 学生数为：" + str(len(train_datas)))
        # 训练集学生的数量
        length = len(train_datas)
        #######################################################放入模型训练#############################################################
        # 2. 准备训练参数
        # 3. 放入训练模型（类）
        Train = Train_deap(winId, WindowsNum, survival, max_generations, train_datas, train_steps, kc_nums, length, print_str)

        # 训练
        model_param = Train.run(sourcePath, winId, dataName, foldName, LastGenPop)
        paramList.append(model_param)
    ########################################################测试#########################################################################
    Testlength = len(All_test_steps)
    testModelParam = paramList[-1]

    auc, bb_acc, Precision, Recall, F1, rmse, APScore = Train.Test(0, testModelParam, All_test_datas, All_test_steps, Testlength, dataName, foldName)
    # Train = None

    # 保存所有的参数列表
    mytools.Save_2dimList(paramList,
                          './CEAtestResults/' + dataName + '/' + foldName + '/forget' + str(WindowsNum) + 'runs' + str(
                              survival) + '/paramList.txt')

    return auc, bb_acc, Precision, Recall, F1, rmse, APScore


if __name__ == '__main__':

    time_start = time.time()
    print_str = "\n###############窗口5；测试algebra06数据集*1折*1轮（80代，种群规模100）：只跑1折1轮,代理模型每5代训练1次,未评估的个体跟基线比超过0.5再评估，去除重复添加的个体，局部搜索噪声0.01（之前是0.001##################\n"
    # print_str = "\n#########################只在前2个窗口运行45次局部搜索，后面只运行一次###########################\n"
    # 设置训练的遗传代数
    max_generations = 80     # 80

    SoursePath = 'DataSets'
    dataName = 'MultiAssist09'  # 168
    # dataName = 'algebra06'  # 112
    # dataName = 'CEAtestResults'
    # dataName = 'algebra06_kcs'
    foldList = ['fold0']

    WinNumList = [5]
    # survivalList = [0.1, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    runs = [5]
    for WinN in WinNumList:
        # for s in survivalList:
        for fold in foldList:
            writeData = []
            for r in runs:
                auc, bb_acc, Precision, Recall, F1, rmse, APScore = single_run((SoursePath, dataName, fold, 168, max_generations, WinN, r, print_str))
                # auc, bb_acc, Precision, Recall, F1, APScore = 1, 2 , 3, 4, 5, 6
                writeData.append((auc, bb_acc, Precision, Recall, F1, rmse, APScore))
            if not os.path.exists('./CEAtestResults/' + dataName + '/' + fold):
                mkdir('./CEAtestResults/' + dataName + '/' + fold)
            with open('./CEAtestResults/' + dataName + '/' + fold + '/' + 'testResult.csv', 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(writeData)

    time_end = time.time()

    dirTmp = './CEAtestResults/' + dataName
    with open(dirTmp + '/' + 'time_all.csv', 'a', newline='', encoding='utf-8') as f:
        f.write(print_str)
        f.write('运行总时间：'+ str(time_end - time_start))

