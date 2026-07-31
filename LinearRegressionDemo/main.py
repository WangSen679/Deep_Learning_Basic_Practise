import torch

# 1. 产生一些模拟输入数据 x，形状为 (100, 1)，即 100 个样本，每个样本 1 个特征
# 使用 torch.rand 创建 [0, 1) 区间的随机数 (参考课件第 26 页)
x = torch.rand(100, 1)

# 2. 设定真实的规律：y = 3x + 2，并加上一点随机噪声（模拟真实世界中不完美的数据）
# 目标是让神经网络通过学习，自动找出权重 3 和偏置 2
noise = torch.randn(100, 1) * 0.1  # 均值为0，方差为0.1的噪声
y = 3 * x + 2 + noise

print("输入数据 x 的前5个数据:\n", x[:5])
print("对应标签 y 的前5个数据:\n", y[:5])

import torch.nn as nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        # nn.Linear 内部会自动创建 requires_grad=True 的权重 weight 和偏置 bias
        # 输入维度为 1，输出维度为 1 (参考课件第 77 页、91 页)
        self.linear = nn.Linear(in_features=1, out_features=1)
        
    def forward(self, x):
        # 定义前向传播过程 (参考课件第 72-73 页)
        return self.linear(x)

# 实例化模型
model = LinearRegressionModel()

import torch.optim as optim

# 1. 定义均方误差损失函数 (参考课件第 116 页)
criterion = nn.MSELoss()

# 2. 定义 SGD 优化器，学习率设为 0.1 (参考课件第 124 页)
# 我们将模型的参数 model.parameters() 传给优化器，让它负责更新这些参数
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 训练 1000 轮
for epoch in range(1001):
    # 1. 梯度清零
    optimizer.zero_grad()
    
    # 2. 前向传播：得到预测结果
    y_pred = model(x)
    
    # 3. 计算损失值
    loss = criterion(y_pred, y)
    
    # 4. 反向传播：计算梯度
    loss.backward()
    
    # 5. 更新参数
    optimizer.step()
    
    # 每 10 轮打印一次训练状态，观察损失值是否在减小
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# 打印模型学到的状态字典 (参考课件第 125-126 页)
learned_parameters = model.state_dict()

# 提取权重和偏置
weight = learned_parameters['linear.weight'].item()
bias = learned_parameters['linear.bias'].item()

print("\n--- 训练结果分析 ---")
print(f"神经网络学到的 权重(W): {weight:.4f} (期望接近 3)")
print(f"神经网络学到的 偏置(B): {bias:.4f}  (期望接近 2)")