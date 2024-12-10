import numpy as np
import igraph
import linecache

# 读txt文件里面的数据转化为二维列表
def Read_2dimList(filename):
    file1 = open(filename, "r")
    list_row =file1.readlines()
    list_source = []
    for i in range(len(list_row)):
        column_list = list_row[i].strip().split("\t")  # 每一行split后是一个列表
        list_source.append(column_list)                # 在末尾追加到list_source
    file1.close()
    return list_source


# 读txt文件里面的数据转化为一维列表
def Read_List(filename):
    file1 = open(filename, "r")
    list_row =file1.readline()
    list_source = list_row.strip().split("\t")  # 每一行split后是一个列表
    file1.close()
    return list_source

#保存二维列表到txt文件
def Save_2dimList(list1, filename):
    file2 = open(filename, 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file2.write(str(list1[i][j]))              # write函数不能写int类型的参数，所以使用str()转化
            file2.write('\t')                          # 相当于Tab一下，换一个单元格
        file2.write('\n')                              # 写完一行立马换行
    file2.close()

#保存一维列表到txt文件
def Save_List(list1, filename):
    file2 = open(filename, 'w')
    for i in range(len(list1)):
        file2.write(str(list1[i]))                  # write函数不能写int类型的参数，所以使用str()转化
        file2.write('\t')                           # 相当于Tab一下，换一个单元格
    file2.close()

def GetG(g_k, q):
    # reshape 成 (1*k)*(k*1)，利用矩阵乘法
    q_transpose = np.transpose(q)
    res = np.dot(g_k, q_transpose)
    row_num = g_k.shape[0]
    res = res.reshape(row_num, 1)
    return 1.0 - 1.0/(1+np.exp(-res))

# a = np.array([[1, 2, 3],
#               [2, 3, 4]
#               ])
# b = np.array([3, 4, 5])
# # print(a.shape)
# c = GetG(a,b)
# print(c.shape[1])
# print(c)


def GetS(s_k, q):
    q_transpose = np.transpose(q)
    res = np.dot(s_k, q_transpose)
    row_num = s_k.shape[0]
    res = res.reshape(row_num, 1)
    return 1.0/(1+np.exp(-res))


def GetG_deap(g_k, q):
    q_transpose = np.transpose(q)
    res = np.dot(g_k, q_transpose)
    return 1.0 - 1.0/(1+np.exp(-res))
    # return 1.0 / (1 + np.exp(-res))


def GetS_deap(s_k, q):
    q_transpose = np.transpose(q)
    res = np.dot(s_k, q_transpose)
    return 1.0/(1+np.exp(-res))


def GetG_deap_weight(g_k, q, g_weight):
    q = q * g_weight
    # q_transpose = np.transpose(q)
    # res = np.dot(g_k, q_transpose)
    res = q * g_k
    # print(np.sum(res))
    return np.sum(res)
    # return 1.0 / (1 + np.exp(-res))


def GetS_deap_weight(s_k, q, s_weight):
    q = q * s_weight
    # q_transpose = np.transpose(q)
    # res = np.dot(s_k, q_transpose)
    res = q * s_k
    # print(np.sum(res))
    return np.sum(res)


def GetG_deap_weight2(g_k, q, g_weight):
    sum_weight = np.sum(g_weight)
    s_weight = g_weight / sum_weight
    res = s_weight * q * g_k
    tmp = s_weight * g_k
    # res = res * g_k
    return (np.sum(res) / np.sum(tmp))

def GetS_deap_weight2(s_k, q, s_weight):
    sum_weight = np.sum(s_weight)
    s_weight = s_weight / sum_weight
    res = s_weight * q * s_k
    tmp = s_weight * s_k
    # res = res * s_k
    return (np.sum(res) / np.sum(tmp))

# 输入：待求种群、汉明距离、个体的基因长度
'''
是一个NBC算法，会对初始种群进行一个聚类的操作，
把一个初始种群聚类为几个小种群，再分别对每个小种群进行交叉变异等操作。
目的是能扩大搜索，找到更多相对最优解
'''


def getMultiPopList(invalid_ind, disMatrix, GENE_LENGTH):
    # fitnessesList = [ind.fitness.values[0] for ind in invalid_ind]  # 各个个体的适应度
    fitnessesList = [ind.fitness.values for ind in invalid_ind]  # 各个个体的适应度
    indDict = dict(zip(range(len(invalid_ind)), fitnessesList))  # 字典：个体序号--适应度
    indDict = dict(sorted(indDict.items(), key=lambda x: x[1], reverse=True))  # 适应度排序
    # print("种群内序号-适应度字典",indDict)  # 排序后的字典{89: 292.0, 30: 286.0, 67: 286.0, 56: 284.0, ......
    sortInd = [i for i in indDict]  # 一维列表 适应度排序的序号
    # print("种群内排名",sortInd)  # 排序后的序号[89, 30, 67, 56, ......

    g = igraph.Graph(directed=True)
    g.add_vertices(sortInd)  # 添加这么多顶点
    g.vs['label'] = sortInd  # 顶点添加标签 以sortInd列表内容（即序号）为标签
    g.es['weight'] = 1.0  # 边长度

    # 算法3当中的的第一个for循环构建整个图 ###############
    index = 0
    weightEdgesDict = {}

    for i in sortInd[1:]:
        newsortInd = sortInd[index + 1:]  # 第一次执行时从下标1开始
        # print(newsortInd)
        idisListTemp = disMatrix[i]
        # j为不会比当前节点更好的节点，把他们对应的值置为最大
        for j in newsortInd:
            idisListTemp[j] = GENE_LENGTH + 1
        # print(idisListTemp)
        # minDis返回的是最小的距离的下标，也就是要连接的节点编号
        minDisIndex = idisListTemp.index(min(idisListTemp))
        minDis = idisListTemp[minDisIndex]
        # print('最小距离的下标', minDisIndex)
        # print('最小距离', minDis)
        # 找到节点编号对应的下标
        # print('i', i)
        # print('minDisindex', minDisIndex)
        nodeIdSource = sortInd.index(i)
        nodeIdTarget = sortInd.index(minDisIndex)
        # nodeIdSource = i
        # nodeIdTarget = minDisIndex
        # print('nodeIdSource', nodeIdSource)
        # print('nodeIdTarget', nodeIdTarget)

        # 而g的add_edge方法建立边是根据下标建立的
        g.add_edge(nodeIdSource, nodeIdTarget)
        if (minDis == 0):
            g[nodeIdSource, nodeIdTarget] = 1
        else:
            g[nodeIdSource, nodeIdTarget] = minDis
        # # weightEdgesDIct是按节点下标存的
        # weightEdgesDict[(nodeIdSource, nodeIdTarget)] = minDis

        # weightEdgesDIct按真实编号存的
        weightEdgesDict[(i, minDisIndex)] = minDis
        # igraph.plot(g)
        index += 1
    # print(g.es['weight'])
    # 计算meanDIs
    meanDis = sum(g.es['weight']) / len(g.es['weight'])
    # print('meanDis', meanDis)
    # 计算度
    numbers = g.indegree()
    # print(numbers)
    # neighbors存了每个节点的度，是按真实节点的名称存的
    neighbors = dict(zip(g.vs['label'], numbers))

    # print("weightEdgesDict",weightEdgesDict)
    # # print(len(weightEdgesDict))
    # print("neighbors",neighbors)
    # igraph.plot(g)
    # 删除边：删除时是按下标去删除的
    for node in weightEdgesDict:
        # print(node)
        # print(weightEdgesDict[node])
        # ni = neighbors[sortInd[node[0]]]
        ni = neighbors[node[0]]
        # print('ni',ni)
        nodeIdSource = sortInd.index(node[0])
        nodeIdTarget = sortInd.index(node[1])
        if (weightEdgesDict[node] > meanDis and ni > 2):  # and ni > 10
            # print(node)
            g.delete_edges((nodeIdSource, nodeIdTarget))

    # igraph.plot(g).save('hahaah.png')
    newg = g.as_undirected().decompose()
    return newg

# 汉明距离 码距 参数是种群
# 种群实质是个二维列表 里面每个列表之间两两计算码距
def hammingDis(invalid_ind):
    Matrix = []  # 二维列表
    for i in invalid_ind:
        sonMatrix = []
        for j in invalid_ind:
            dis = sum([abs(ch1 - ch2) for ch1, ch2 in zip(i, j)])
            sonMatrix.append(dis)
        Matrix.append(sonMatrix)
    return Matrix

def local_search_train_0():
    return 0

# 获取数据集文件中最长时间步的学生，未来窗口划分使用
def getMaxLen(folder):
    lineNum = 1
    maxLen = -1
    while True:
        nowLen = linecache.getline(folder, lineNum).strip()
        if not nowLen:
            break
        nowLen = int(nowLen)
        maxLen = nowLen if nowLen > maxLen else maxLen
        lineNum = lineNum + 3
    linecache.clearcache()
    return maxLen

# 读取指定窗口格的数据
def Read_window(folder, windowSize, windowID):
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
    return Datas, Steps
