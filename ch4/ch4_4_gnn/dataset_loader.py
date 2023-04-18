# ch4/ch4_4_gnn/dataset_loader.py
# 第三方库
import numpy as np
import pandas as pd
from torch_geometric_temporal.signal import StaticGraphTemporalSignal


class GraphDatasetLoader(object):
    """时空图数据集加载器
    """
    # 初始化

    def __init__(self, data_type, path_data, path_graph):
        """时空图数据集加载器初始化

        参数:
            data_type (string): 变量类型(aqi或traffic).
            path_data (string): 节点特征数据文件路径.
            path_graph (string): 图拓扑结构文件路径.
        """
        self._read_local_data(data_type, path_data, path_graph)

    def __len__(self):
        """获取数据集长度

        返回值:
            int: 数据集长度.
        """
        return len(self.features)

    def _get_edges(self):
        """获取图中的边
        """
        self._edges = np.array(
            self._dataset["edges"]).T  # [2, num_edge]

    def _get_edge_weights(self):
        """获取图中边的权重
        """
        self._edge_weights = np.array(
            self._dataset["weights"]).T  # [num_edge, ]

    def _get_targets_and_features(self, H, num_train):
        """获取特征和标签

        参数:
            H (int): 输入历史数据长度.
            num_train (int): 用于构成训练集的序列长度.
        """
        # 归一化
        targets = np.array(self._dataset["x"])  # [L-H, N]
        self.min = np.min(targets[:num_train-H], axis=0)  # [N, ]
        self.max = np.max(targets[num_train-H:], axis=0)  # [N, ]
        targets = (targets - self.min) / (self.max - self.min)  # [L-H, N]

        # 构建特征 [[N, 1, H], [N, 1, H], ...]
        self.features = [np.expand_dims(targets[i:i+self.H, :].T, axis=1)
                         for i in range(targets.shape[0]-self.H)]

        # 构建标签 [[N, 1], [N, 1], ...]
        self.targets = [np.expand_dims(targets[i+self.H, :].T, axis=1)
                        for i in range(targets.shape[0]-self.H)]

    def _read_local_data(self, data_type, path_data, path_graph):
        """读取节点特征数据文件

        参数:
            data_type (string): 变量类型(aqi或traffic).
            path_data (string): 节点特征数据文件路径.
            path_graph (string): 图拓扑结构文件路径.
        """
        if data_type == 'aqi':
            info = ['datetime', 'type']
        elif data_type == 'traffic':
            info = ['id']
        else:
            print('错误数据类型')
            return

        # 读取节点特征
        df = pd.read_csv(path_data)
        nodes_names = df.columns[len(info):]
        self.x = df[nodes_names].values  # [L, N]

        # 读取邻接矩阵
        edges = []
        weights = []
        data = pd.read_csv(path_graph)
        for _, edge in data.iterrows():
            start, end, _ = int(edge['start']), int(edge['end']), edge['dist']
            edges.extend([[start, end], [end, start]])  # 同时加入双向两条边
            weights.extend([1, 1])  # 设置双向两条边的权重均为1

        # 构建原始数据集
        self._dataset = {"x": self.x, "edges": edges, "weights": weights}

    def get_dataset(self, H, num_train, num_test):
        """获取数据集

        参数:
            H (int): 历史数据长度.
            num_train (int): 训练集样本数量.
            num_test (int): 测试集样本数量.

        返回值:
            StaticGraphTemporalSignal: 静态图时间信号数据集
        """
        self.H = H
        self.num_train = num_train
        self.num_test = num_test
        self._get_edges()
        self._get_edge_weights()
        self._get_targets_and_features(H, num_train)
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.features, self.targets)
        return dataset

    def train_test_split(self, dataset):
        """训练测试集划分

        参数:
            dataset (StaticGraphTemporalSignal): 静态图时间信号数据集.

        返回值:
            StaticGraphTemporalSignal: 训练集和测试集
        """
        return temporal_signal_split(
            dataset, self.num_train, self.num_test, self.H)


def temporal_signal_split(data_iterator, num_train, num_test, H):
    """训练测试样本划分

    参数:
        data_iterator (StaticGraphTemporalSignal): 静态图时间信号数据集.
        num_train (int): 训练集样本数量.
        num_test (int): 测试集样本数量.
        H (int): 历史数据长度.

    返回值:
        StaticGraphTemporalSignal: 训练集和测试集
    """
    train_snapshots = num_train - H
    test_snapshots = train_snapshots + num_test

    train_iterator = StaticGraphTemporalSignal(
        data_iterator.edge_index,
        data_iterator.edge_weight,
        data_iterator.features[0:train_snapshots],
        data_iterator.targets[0:train_snapshots],
    )

    test_iterator = StaticGraphTemporalSignal(
        data_iterator.edge_index,
        data_iterator.edge_weight,
        data_iterator.features[train_snapshots:test_snapshots],
        data_iterator.targets[train_snapshots:test_snapshots],
    )

    return train_iterator, test_iterator


if __name__ == '__main__':

    # 数据读取
    loader = GraphDatasetLoader(
        data_type='aqi',
        path_data='./data/data_aqi.csv',
        path_graph='./data/graph_aqi.csv'
    )

    # 创建数据集
    H = 12
    dataset = loader.get_dataset(
        H,
        num_train=7*24*3,  # 7天/周 × 24时/天 × 3周
        num_test=7*24*1  # 7天/周 × 24时/天 × 1周
    )

    # 训练测试划分
    train, test = loader.train_test_split(dataset)
    print(f'{train.snapshot_count=}, {test.snapshot_count=}')

    # 获取一个训练样本
    snapshot = next(iter(train))

    # 获取数据
    targets = snapshot.y
    inputs = snapshot.x
    inputs_edge = snapshot.edge_index
    print(f'{targets.shape=}, {inputs.shape=}, {inputs_edge.shape}')
