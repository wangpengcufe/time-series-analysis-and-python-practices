# utils/dataset.py
# 第三方库
import pandas as pd
from IPython.display import display


def stats(series):
    """计算单变量时间序列统计量

    参数:
        series (1d numpy array): 时间序列数据.
    """
    series = pd.DataFrame(series)
    stat = pd.DataFrame(
        data=[[
            series.shape[0],
            series.max().values[0],
            series.min().values[0],
            series.mean().values[0],
            series.std(ddof=1).values[0],  # 样本标准差
            series.skew().values[0],
            series.kurt().values[0]
        ]],
        columns=[
            '序列长度', '最大值', '最小值', '均值', '标准差', '偏度', '峰度'
        ]).round(2)

    # 打印统计结果,可替换为print()
    display(stat)


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """将时间序列数据转换为监督学习数据集

    参数:
        data (list or 2d numpy array): 时间序列数据.
        n_in (int, optional): 输入变量X的滞后数目. 默认为 1.
        n_out (int, optional): 输出变量y的超前数目. 默认为 1.
        dropnan (bool, optional): 是否去除带有空值的行. 默认为 True.

    返回值:
        pandas dataframe: 监督学习数据集
    """
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()

    # 输入序列 (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # 输出序列 (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # 输入/输出拼接
    dataset = pd.concat(cols, axis=1)
    dataset.columns = names

    # 删除空值行
    if dropnan:
        dataset.dropna(inplace=True)

    # 重置行索引
    dataset.reset_index(inplace=True, drop=True)

    return dataset
