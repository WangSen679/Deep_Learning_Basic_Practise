# 时序预测：Conv1D vs LSTM

使用 PyTorch 实现一维卷积（Conv1D）和长短期记忆网络（LSTM）对带噪声的正弦函数进行时序预测，并对比两种模型的效果。

## 环境要求

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

## 环境配置

### 1. 使用 Conda 创建环境（推荐）

```bash
# 创建 conda 环境
conda create -n torch_ts python=3.10

# 激活环境
conda activate torch_ts

# 安装核心依赖
conda install numpy matplotlib scikit-learn
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# 安装 Jupyter 内核
conda install ipykernel jupyter
python -m ipykernel install --user --name torch_ts --display-name "Python (torch_ts)"
```

### 2. 检查 PyTorch 安装

```python
import torch
print(torch.__version__)       # 应输出 2.x
print(torch.cuda.is_available())  # True 表示 GPU 可用
```

### 3. Jupyter Notebook 内核配置

本项目的 Notebook 使用了 `%matplotlib inline`，在 Jupyter 中可直接运行。如需选择正确的内核：

```bash
# 查看已安装的内核列表
jupyter kernelspec list

# 输出示例：
# Available kernels:
#   torch_ts              C:\Users\xxx\AppData\Roaming\jupyter\kernels\torch_ts
#   python3               ...\share\jupyter\kernels\python3
```

打开 Notebook 后，在 VS Code 右上角 Kernel 选择器中选择 `Python (torch_ts)` 或你创建的环境名称。

### 4. 验证环境

运行以下命令确认所有依赖已就绪：

```bash
python -c "
import numpy
import matplotlib
import torch
import sklearn
print('NumPy:', numpy.__version__)
print('Matplotlib:', matplotlib.__version__)
print('PyTorch:', torch.__version__)
print('scikit-learn:', sklearn.__version__)
"
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `时序预测_Conv1D_LSTM.ipynb` | Jupyter Notebook（交互式运行，含图文说明） |
| `时序预测_Conv1D_LSTM.py` | Python 脚本（直接运行） |
| `*.png` | 训练结果可视化图片 |

## 运行方式

### Notebook（推荐）

在 VS Code 中打开 `时序预测_Conv1D_LSTM.ipynb`，选择内核后逐 Cell 运行。

### Python 脚本

```bash
python 时序预测_Conv1D_LSTM.py
```

## 结果示例

| 模型 | 测试集 MSE |
|------|-----------|
| Conv1D | 0.067 |
| LSTM | 0.046 |

LSTM 在本任务中表现优于 Conv1D，预测曲线更贴近真实值。
