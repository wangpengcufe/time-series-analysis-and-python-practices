# utils/plot.py
# 标准库
import platform

# 第三方库
import numpy as np
import matplotlib.pyplot as plt

# 自定义模块
from .metrics import all_metrics


def set_matplotlib(plot_dpi=80, save_dpi=600, font_size=12):
    """配置matplotlib全局绘图参数

    参数:
        plot_dpi (int, optional): 绘图dpi. 默认为 80.
        save_dpi (int, optional): 保存图像dpi. 默认为 600.
        font_size (int, optional): 字号. 默认为 12.
    """
    # 中文字体设置
    sys = platform.system()
    if sys == 'Linux':
        plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']
    elif sys == 'Windows':
        plt.rcParams['font.sans-serif'] = ['SimHei']
    elif sys == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    else:
        print('中文字体设置失败')

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = font_size

    # 图像分辨率设置
    plt.rcParams['figure.dpi'] = plot_dpi
    plt.rcParams['savefig.dpi'] = save_dpi


def plot_dataset(
        train, test, size=(6, 3.5), xlabel='', ylabel='', fig_name=''):
    """绘制时间序列数据集及其训练验证划分图像

    参数:
        train (1d numpy array): 用于构成训练集的序列.
        test (1d numpy array): 用于构成测试集的序列.
        size (tuple, optional): 图像尺寸. 默认为 (6, 3.5).
        xlabel (str, optional): x轴标签. 默认为 ''.
        ylabel (str, optional): y轴标签. 默认为 ''.
        fig_name (str, optional): 图像名. 默认为 ''.
    """
    # 横轴范围计算
    x_train = np.linspace(1, len(train), len(train))
    x_test = np.linspace(len(train), len(train)+len(test), len(test)+1)

    # 绘图
    plt.figure(figsize=size)
    plt.plot(x_train, train, label='训练集')
    plt.plot(x_test, np.append(train[-1], test), label='测试集')

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'./fig/{fig_name}.jpg', bbox_inches='tight')
    plt.show()


def plot_decomposition(
        series, decomposition, size=(6, 7),
        xlabel='', ylabel='', fig_name=''):
    """绘制原始序列及其季节性分量

    参数:
        series (1d numpy array): 原始序列/被分解的序列
        decomposition (DecomposeResult): 分解结果对象
        size (tuple, optional): 图像尺寸. 默认为 (6, 7).
        xlabel (str, optional): x轴标签. 默认为 ''.
        ylabel (str, optional): y轴标签. 默认为 ''.
        fig_name (str, optional): 图像名. 默认为 ''.
    """
    # 绘图
    plt.figure(figsize=size)

    # 原始数据
    plt.subplot(411)
    plt.plot(series, label='原始数据')
    plt.legend(loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 趋势项
    plt.subplot(412)
    plt.plot(decomposition.trend, color='r', label='趋势项')
    plt.legend(loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 季节项
    plt.subplot(413)
    plt.plot(decomposition.seasonal, color='g', label='季节项')
    plt.legend(loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # 残差
    plt.subplot(414)
    plt.plot(decomposition.resid, color='b', label='残差项')
    plt.legend(loc='upper left')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.savefig(f'./fig/{fig_name}.jpg', bbox_inches='tight')
    plt.show()


def plot_losses(
        train_loss, val_loss=None, size=(6, 3.5),
        xlabel='', ylabel='', fig_name=''):
    """绘制模型训练损失和验证损失图像

    参数:
        train_loss (1d numpy array): 模型训练损失.
        val_loss (1d numpy array, optional): 模型测试损失. 默认为 None.
        size (tuple, optional): 图像尺寸. 默认为 (6, 3.5).
        xlabel (str, optional): x轴标签. 默认为 ''.
        ylabel (str, optional): y轴标签. 默认为 ''.
        fig_name (str, optional): 图像名. 默认为 ''.
    """
    plt.figure(figsize=size)
    plt.plot(train_loss, label='训练损失')
    if val_loss:
        plt.plot(val_loss, label='验证损失')

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'./fig/{fig_name}.jpg', bbox_inches='tight')
    plt.show()


def plot_results(
        y_true, y_pred, size=(6, 3.5), xlabel='', ylabel='', fig_name=''):
    """绘制预测结果曲线图像

    参数:
        y_true (1d numpy array): 观测值/真值.
        y_pred (1d numpy array): 预测值.
        size (tuple, optional): 图像尺寸. 默认为 (6, 3.5).
        xlabel (str, optional): x轴标签. 默认为 ''.
        ylabel (str, optional): y轴标签. 默认为 ''.
        fig_name (str, optional): 图像名. 默认为 ''.
    """
    plt.figure(figsize=size)
    plt.plot(y_true.squeeze(), label='观测值')
    plt.plot(y_pred.squeeze(), label='预测值')

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'./fig/{fig_name}.jpg', bbox_inches='tight')
    plt.show()


def plot_parity(
        y_true, y_pred, size=(6, 3.5), xlabel='', ylabel='', fig_name=''):
    """绘制预测结果Parity Plot图像

    参数:
        y_true (1d numpy array): 观测值/真值.
        y_pred (1d numpy array): 预测值.
        size (tuple, optional): 图像尺寸. 默认为 (6, 3.5).
        xlabel (str, optional): x轴标签. 默认为 ''.
        ylabel (str, optional): y轴标签. 默认为 ''.
        fig_name (str, optional): 图像名. 默认为 ''.
    """
    x = y_true
    y = y_pred

    # 图像边界计算
    bounds = (
        min(x.min(), x.min()) - int(0.1 * x.min()),
        max(x.max(), x.max()) + int(0.1 * x.max())
    )

    # 绘图
    plt.figure(figsize=size)
    ax = plt.gca()
    ax.plot(x, y, '.', label='观测-预测')
    ax.plot([0, 1], [0, 1], lw=2, alpha=1.0,
            transform=ax.transAxes, label='$y=x$')

    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'./fig/{fig_name}.jpg', bbox_inches='tight')
    plt.show()


def plot_metrics_distribution(
        y_true, y_pred, size=(10, 3), xlabel='', ylabel='', fig_name=''):
    """绘制各节点预测误差分布图像

    参数:
        y_true (2d numpy array): 观测值/真值.
        y_pred (2d numpy array): 预测值.
        size (tuple, optional): 图像尺寸. 默认为 (6, 3.5).
        xlabel (str, optional): x轴标签. 默认为 ''.
        ylabel (str, optional): y轴标签. 默认为 ''.
        fig_name (str, optional): 图像名. 默认为 ''.
    """
    all_rmse = []
    all_mae = []
    all_sde = []

    # 各节点预测结果误差指标计算
    N = y_true.shape[1]
    for idx_node in range(N):
        metric_value = all_metrics(
            y_true[:, idx_node],
            y_pred[:, idx_node],
            return_metrics=True)
        all_rmse.append(metric_value['rmse'])
        all_mae.append(metric_value['mae'])
        all_sde.append(metric_value['sde'])

    # 绘图
    plt.figure(figsize=size)
    plt.bar(
        x=np.arange(N),
        height=y_true.mean(axis=0).squeeze(),
        color='lightgray',
        label='Mean')
    plt.plot(all_rmse, 'v--', label='RMSE')
    plt.plot(all_mae,  's--', label='MAE')
    plt.plot(all_sde,  'd--', label='SDE')

    plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f'./fig/{fig_name}.jpg', bbox_inches='tight')
    plt.show()
