# utils/metrics.py
# 第三方库
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import r2_score as r2
from scipy.stats import pearsonr


def rmse(y_true, y_pred):
    """计算均方根误差RMSE

    参数:
        y_true (1d numpy array): 观测值/真值.
        y_pred (1d numpy array): 预测值.

    返回值:
        float: 均方根误差RMSE
    """
    return np.sqrt(mse(y_true, y_pred))


def sde(y_true, y_pred):
    """计算误差标准差SDE

    参数:
        y_true (1d numpy array): 观测值/真值.
        y_pred (1d numpy array): 预测值.

    返回值:
        float: 误差标准差SDE
    """
    return np.std(y_true - y_pred)


def pcc(y_true, y_pred):
    """计算皮尔森相关系数PCC

    参数:
        y_true (1d numpy array): 观测值/真值.
        y_pred (1d numpy array): 预测值.

    返回值:
        float: 皮尔森相关系数PCC
    """
    return pearsonr(y_true, y_pred)[0]


def all_metrics(y_true, y_pred, return_metrics=False):
    """返回或打印全部误差评价指标

    参数:
        y_true (1d numpy array): 观测值/真值.
        y_pred (1d numpy array): 预测值.
        return_metrics (bool, optional): 是否返回指标变量. 默认为 False.

    返回值:
        dict: 由全部误差评价指标构成的字典
    """

    # 数据压缩
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    # 模型评价
    metrics = {
        'mse': mse(y_true, y_pred),
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true, y_pred)*100,
        'sde': sde(y_true, y_pred),
        'r2': r2(y_true, y_pred),
        'pcc': pcc(y_true, y_pred)
    }

    # 输出结果
    if return_metrics:
        return metrics
    else:
        print(f"mse={metrics['mse']:.3f}")
        print(f"rmse={metrics['rmse']:.3f}")
        print(f"mae={metrics['mae']:.3f}")
        print(f"mape={metrics['mape']:.3f}%")
        print(f"sde={metrics['sde']:.3f}")
        print(f"r2={metrics['r2']:.3f}")
        print(f"pcc={metrics['pcc']:.3f}")
