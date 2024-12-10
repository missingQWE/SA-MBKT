import torch
import os
import sys
import logging
import torch.utils.data as data

# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "cpu"

class CachedSubset(data.Dataset):
    r"""
    Subset of a dataset at specified indices.

    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

        self.cache = [self.dataset[i] for i in self.indices]

    def __getitem__(self, idx):
        return self.cache[idx]

    def __len__(self):
        return len(self.indices)

def shuffle(*items):
    example, _ = items
    batch_size, _ = example.size()
    index = torch.randperm(batch_size, device=example.device)

    return [item[index] for item in items]

def to_device(*items):
    return [item.to(device=device) for item in items]

def list_select(items, index):
    if isinstance(index, torch.Tensor):
        index = index.tolist()
    print("tempPop[0]:", items[0])
    print("index:", index, type(index))
    return [items[i] for i in index]

def concat(a, b):
    return [torch.cat([item0, item1]) for item0, item1 in zip(a, b)]

'''计算肯德尔相关系数'''
def compute_kendall_tau_AR(ranker, archs, performances):
    '''
    Kendall Tau is a metric to measure the ordinal association between two measured quantities.
    Refer to https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient
    '''
    # assert len(matrix) == len(ops) == len(performances), "Sequence a and b should have the same length while computing kendall tau."
    length = len(performances)
    count = 0
    total = 0

    archs = transpose_l([torch.unbind(item) for item in archs])
    outputs = cartesian_traverse(archs, archs, ranker, up_triangular=True)

    p_combination = _sign((outputs-0.5).cpu().tolist())

    for i in range(length-1):
        for j in range(i+1, length):
            count += p_combination[total] * _sign(performances[i]-performances[j])
            total += 1

    assert len(p_combination) == total
    Ktau = count / total
    return Ktau

'''转置'''
def transpose_l(items):
    return list(map(list, zip(*items)))

'''用比较器计算两个架构列表中每个架构对的相对评分'''
def cartesian_traverse(arch0, arch1, ranker, up_triangular=False):
    m, n = len(arch0), len(arch1)
    outputs = []
    with torch.no_grad():
        for index in index_generate(m, n, up_triangular):
            i, j = transpose_l(index)
            # 选择并批量化arch0和arch1中的架构
            a = batchify(select(arch0, i))
            b = batchify(select(arch1, j))
            # 使用ranker计算输出
            output = ranker(a, b)
            outputs.append(output)
    outputs = torch.cat(outputs, dim=0)
    # 返回输出，根据up_triangular决定返回形式
    if up_triangular:
        return outputs
    else:
        return outputs.view(m, n)   # 返回m*n的张量

'''生成矩阵索引对'''
def index_generate(m, n, up_triangular=False, max_batch_size=1024):
    # 生成上三角矩阵索引对
    if up_triangular:
        indexs = []
        for i in range(m-1):
            for j in range(i+1, n):
                indexs.append((i, j))
                if len(indexs) == max_batch_size:
                    yield indexs
                    indexs = []
        if indexs:
            yield indexs
    # 生成完整矩阵索引对
    else:
        indexs = []
        for i in range(m):
            for j in range(n):
                indexs.append((i, j))
                if len(indexs) == max_batch_size:
                    yield indexs
                    indexs = []
        if indexs:
            yield indexs

def _sign(number):
    if isinstance(number, (list, tuple)):
        return [_sign(v) for v in number]
    if number >= 0.0:
        return 1
    elif number < 0.0:
        return -1

def select(items, index):
    return [items[i] for i in index]

def batchify(items):
    if isinstance(items[0], (list, tuple)):
        transposed_items = transpose_l(items)
        return [torch.stack(item, dim=0) for item in transposed_items]
    else:
        return torch.stack(items, dim=0)

def get_logger(name: str, output_directory: str, log_name: str, debug: str) -> logging.Logger:
    logger = logging.getLogger(name)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s: %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_directory is not None:
        file_handler = logging.FileHandler(os.path.join(output_directory, log_name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    logger.propagate = False
    return logger


# logger = get_logger(name="project", output_directory="ProxyLog", log_name="log1-loss.txt", debug=False)