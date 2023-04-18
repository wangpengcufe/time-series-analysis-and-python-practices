# ch4/ch4_5_att/dataset.py
# 第三方库
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TimeDataset(Dataset):
    """时间序列数据集
    """

    def __init__(
        self,
        path_data,
        data_type,
        size_window=24,
        size_input=1,
        size_output=1
    ):
        """时间序列数据集初始化

        参数:
            path_data (str): 时间序列数据文件路径.
            data_type (str): 变量类型(列名).
            size_window (int, optional): 滑动窗口大小. 默认为24.
            size_input (int, optional): 输入滑动窗口个数. 默认为1.
            size_output (int, optional): 输出滑动窗口个数. 默认为1.
        """
        # 加载数据
        data = pd.read_csv(path_data)
        data = data[data_type].values
        self.data = data

        # 划分滑动窗口
        data = np.split(  # 由滑动窗口组成的列表, 各窗口不重叠
            data,
            int(len(data) / size_window),
            axis=0
        )

        # 获取数据长度
        size_all = len(data)
        size_group = size_input + size_output

        # 获取数据特征和标签
        inputs = []
        targets = []

        # 遍历构造数据特征和标签
        for i in range(size_all):
            if i + size_group > size_all:
                break
            inputs.append(  # 追加size_input个滑动窗口组成的列表
                data[i:i+size_input])
            targets.append(  # 追加size_output个滑动窗口组成的列表
                data[i+size_input:i+size_input+size_output])

        # 重构为numpy数组
        self.inputs = np.array(inputs)     # [sum_B, size_input, H]
        self.targets = np.array(targets)  # [sum_B, size_output, H]

    def __len__(self):
        """获取数据集长度

        返回值:
            int: 数据集长度
        """
        return len(self.inputs)

    def __getitem__(self, idx):
        """获取一个数据样本

        参数:
            idx (int): 样本索引

        返回值:
            cell: 数据特征及数据标签
        """
        return self.inputs[idx, :, :], self.targets[idx, :, :]


if __name__ == '__main__':

    # 测试
    data = TimeDataset('./data/data_temp.csv', 'AirTemp', 24, 1, 1)
    print(data.inputs.shape)
    print(data.targets.shape)
