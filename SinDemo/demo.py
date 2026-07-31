import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# =====================================================================
# 1. 准备数据（非线性正弦波）
# =====================================================================
# 创建 200 个在 [-3.14, 3.14] 之间均匀分布的点，并增加一维以符合 PyTorch 输入要求 (batch_size, 1)
x = torch.linspace(-3.14, 3.14, 200).unsqueeze(1)
# 目标曲线：y = sin(x) + 随机噪声 (参考课件第26页创建随机数)
noise = torch.randn(x.size()) * 0.15
y = torch.sin(x) + noise

# =====================================================================
# 2. 构建多层神经网络 (参考课件第73页 Module 类与第82页 Sequential 容器)
# =====================================================================
class NonLinearModel(nn.Module):
    def __init__(self):
        super(NonLinearModel, self).__init__()
        # 使用 Sequential 将多个层级联在一起
        # 结构：输入(1) -> 隐藏层1(20) -> 激活函数 -> 隐藏层2(20) -> 激活函数 -> 输出(1)
        self.net = nn.Sequential(
            nn.Linear(1, 20),      # 第一隐藏层
            nn.GELU(),             # 激活函数，使网络具备非线性拟合能力 (参考课件第77页)
            nn.Linear(20, 20),     # 第二隐藏层
            nn.Tanh(),             # 激活函数
            nn.Linear(20, 1)       # 输出层
        )
        
    def forward(self, x):
        return self.net(x)

# 实例化模型
model = NonLinearModel()

# =====================================================================
# 3. 定义损失函数与优化器 (参考课件第116页与第122页)
# =====================================================================
criterion = nn.MSELoss()
# 使用 Adam 优化器，它在处理非线性复杂任务时通常比 SGD 收敛更快
optimizer = optim.Adam(model.parameters(), lr=0.01)

# =====================================================================
# 4. 训练模型 (参考课件第124页模板)
# =====================================================================
epochs = 1000  # 非线性拟合需要更多的训练轮数
for epoch in range(epochs + 1):
    optimizer.zero_grad()       # 1. 梯度清零
    y_pred = model(x)           # 2. 前向传播
    loss = criterion(y_pred, y) # 3. 计算损失
    loss.backward()             # 4. 反向传播计算梯度
    optimizer.step()            # 5. 更新网络参数
    
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.6f}")

# =====================================================================
# 5. 结果可视化 (使用 matplotlib 绘制拟合效果)
# =====================================================================
# 在评估/推理模式下，我们不需要计算梯度，使用 torch.no_grad() 节省资源 (参考课件第119页)
model.eval()
with torch.no_grad():
    y_final_pred = model(x)

plt.figure(figsize=(8, 5))
plt.scatter(x.numpy(), y.numpy(), color='gray', alpha=0.6, label='Real Data (with noise)')
plt.plot(x.numpy(), y_final_pred.numpy(), color='red', linewidth=3, label='Neural Network Fit')
plt.title('Non-linear Fitting using Multi-layer Neural Network')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()