# ch4/ch4_4_gnn/model.py
# 第三方库
import torch
from torch_geometric.nn import GCNConv, GATConv


# 图卷积网络GCN
class GCN(torch.nn.Module):

    def __init__(self, H=12, num_neurons=128):
        """图卷积网络GCN初始化

        参数:
            H (int, optional): 输入历史值长度. 默认为 12.
            num_neurons (int, optional): 神经元数量. 默认为 128.
        """
        super(GCN, self).__init__()
        # GCL层
        self.conv_1 = GCNConv(
            in_channels=H,
            out_channels=num_neurons,
            improved=True,
            add_self_loops=True,
            normalize=True,
            bias=True
        )
        # GCL层
        self.conv_2 = GCNConv(
            in_channels=num_neurons,
            out_channels=num_neurons,
            improved=True,
            add_self_loops=True,
            normalize=True,
            bias=True
        )
        # 线性层/输出层
        self.linear = torch.nn.Linear(
            in_features=num_neurons,
            out_features=1,
            bias=True
        )

    def forward(self, x, edge_index):
        """图卷积网络GCN前向传播

        参数:
            x (torch.Tensor): 节点特征
            edge_index (torch.Tensor): 边

        返回值:
            (torch.Tensor): 图卷积网络GCN前向传播结果
        """
        x = x.squeeze()  # [N, D, H] -> [N, H], D=1 合并D和H所在维度
        y = self.conv_1(x, edge_index)  # [N, H] -> [N, num_neurons]
        y.relu()  # Relu激活

        y = self.conv_2(y, edge_index)  # [N, num_neurons] -> [N, num_neurons]
        y.relu()  # Relu激活

        y = self.linear(y)  # [N, num_neurons] -> [N, 1]
        return y


# 图注意力网络GAT
class GAT(torch.nn.Module):

    def __init__(self, H=12, num_neurons=128, num_heads=3):
        """图注意力网络GAT初始化

        参数:
            H (int, optional): 输入历史值长度. 默认为12.
            num_neurons (int, optional): 神经元数量. 默认为128.
            num_heads (int, optional): 多头注意力数量. 默认为3.
        """
        super(GAT, self).__init__()
        # GAT层
        self.conv_1 = GATConv(
            in_channels=H,
            out_channels=num_neurons,
            heads=num_heads,
            concat=True,
            add_self_loops=True,
            bias=True
        )
        # GAT层
        self.conv_2 = GATConv(
            in_channels=num_neurons*num_heads,
            out_channels=num_neurons,
            heads=num_heads,
            concat=True,
            add_self_loops=True,
            bias=True
        )
        # 线性层/输出层
        self.linear = torch.nn.Linear(
            in_features=num_neurons*num_heads,
            out_features=1,
            bias=True
        )

    def forward(self, x, edge_index):
        """图注意力网络GAT前向传播

        参数:
            x (torch.Tensor): 节点特征
            edge_index (torch.Tensor): 边

        返回值:
            (torch.Tensor): 图注意力网络GAT前向传播结果
        """
        x = x.squeeze()  # [N, D, H] -> [N, H], D=1 合并D和H所在维度
        y = self.conv_1(x, edge_index)  # [N, H] -> [N, num_neurons]
        y.relu()  # Relu激活

        y = self.conv_2(y, edge_index)  # [N, num_neurons] -> [N, num_neurons]
        y.relu()  # Relu激活

        y = self.linear(y)  # [N, num_neurons] -> [N, 1]
        return y


if __name__ == '__main__':

    # GCN模型测试
    model = GCN(12, 128)
    inputs = torch.rand(35, 1, 12)  # [N, D, H]
    inputs_edge = torch.randint(0, 35, size=(2, 926))  # [N, num_edges]
    outputs = model(inputs, inputs_edge)
    print(inputs.shape, outputs.shape)

    # GAT模型测试
    model = GAT(12, 128, 3)
    inputs = torch.rand(35, 1, 12)  # [N, D, H]
    inputs_edge = torch.randint(0, 35, size=(2, 926))  # [N, num_edges]
    outputs = model(inputs, inputs_edge)
    print(inputs.shape, outputs.shape)
