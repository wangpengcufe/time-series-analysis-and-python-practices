# ch4/ch4_5_att/model.py
# 第三方库
import torch
import torch.nn as nn


# 多头注意力网络
class MultiheadsSelfAttention(nn.Module):

    def __init__(self, dim_input, num_neurons, dim_output, num_heads):
        """多头注意力网络初始化

        返回值:
            dim_input (int): 输入数据维度
            num_neurons (int): 神经元数量
            dim_output (int): 输出数据维度
            num_heads (int): 并行多头注意力的数量
        """
        super(MultiheadsSelfAttention, self).__init__()
        # query
        self.query = nn.Linear(
            in_features=dim_input,
            out_features=num_neurons,
            bias=True
        )
        # key
        self.key = nn.Linear(
            in_features=dim_input,
            out_features=num_neurons,
            bias=True
        )
        # value
        self.value = nn.Linear(
            in_features=dim_input,
            out_features=num_neurons,
            bias=True
        )
        # 多头注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=num_neurons,
            num_heads=num_heads,
            bias=True,
            batch_first=True
        )
        # 线性层/输出层
        self.linear = nn.Linear(num_neurons, dim_output) 

    def forward(self, x):
        """多头注意力网络前向传播

        参数:
            x (torch.Tensor): 样本特征

        返回值:
            (torch.Tensor): 多头注意力网络前向传播结果
        """
        # [B, size_input, dim_input] -> [B, size_input, num_neurons]
        q = self.query(x)
        # [B, size_input, dim_input] -> [B, size_input, num_neurons]
        k = self.key(x)
        # [B, size_input, dim_input] -> [B, size_input, num_neurons]
        v = self.value(x)

        output, _ = self.attention(q, k, v)  # [B, size_input, num_neurons]
        output.relu()

        output = self.linear(output)  # [B, size_input, dim_output]

        return output


if __name__ == '__main__':

    # 测试
    model = MultiheadsSelfAttention(24, 48, 24, 8)
    inputs = torch.rand(4, 1, 24)
    outputs = model(inputs)
    print(inputs.shape, outputs.shape)
