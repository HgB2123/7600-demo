# -*- coding: utf-8 -*-
"""
EEG信号识别模型训练脚本
========================
该脚本为ARIN7600项目“Recognition of imagined handwritten content from brain signals”提供了一个完整的基线模型。
它包括以下步骤:
1.  使用PyTorch的Dataset和DataLoader加载和预处理.mat格式的EEG数据。
2.  构建一个适用于EEG信号分类的卷积神经网络 (CNN)。
3.  实现一个完整的训练和验证循环。
4.  在训练过程中监控损失和准确率。

所有参数已通过argparse封装，您可以通过命令行方便地进行调试和修改。
示例运行:
python main.py --epochs 30 --lr 0.0005

请确保您已安装必要的库。为了支持GPU，强烈建议使用Conda安装PyTorch:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install numpy scipy scikit-learn
"""
import os
import argparse  # 导入argparse模块
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import scipy.io
import numpy as np
from sklearn.metrics import accuracy_score


# --- 1. 参数解析 (Argument Parser) ---
def parse_args():
    """
    解析命令行参数。
    """
    parser = argparse.ArgumentParser(description="EEG信号识别模型训练脚本")

    parser.add_argument('--data_path', type=str, default='./data/data_EEG_AI.mat',
                        help='存放.mat数据文件的路径')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='训练和验证时的批处理大小')
    parser.add_argument('--epochs', type=int, default=20,
                        help='训练的总轮次')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='优化器的学习率')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='从总数据集中分割出的验证集比例')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='如果指定，则强制使用CPU进行训练')

    return parser.parse_args()


# --- 2. 自定义数据集类 (Custom Dataset Class) ---
class EEGDataset(Dataset):
    """
    用于加载和预处理EEG .mat文件的PyTorch数据集类。
    """

    def __init__(self, mat_file_path):
        print("正在加载和预处理数据...")
        mat_data = scipy.io.loadmat(mat_file_path)

        # 加载数据和标签
        eeg_data = mat_data['data']
        labels = mat_data['label']

        # 数据预处理
        eeg_data = np.transpose(eeg_data, (2, 0, 1))
        labels = labels.ravel() - 1

        self.X = torch.tensor(eeg_data, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

        self.num_classes = len(np.unique(labels))
        print(f"数据加载完毕。找到 {len(self.X)} 个样本, {self.num_classes} 个类别。")
        print(f"数据形状 (X): {self.X.shape}")
        print(f"标签形状 (y): {self.y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --- 3. 模型架构 (Model Architecture) ---
class SimpleEEGNet(nn.Module):
    """
    一个简化的用于EEG分类的卷积神经网络 (CNN)。
    """

    def __init__(self, num_classes, channels=24, timepoints=801):
        super(SimpleEEGNet, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4)),
            nn.Dropout(0.5),
            nn.Conv2d(16, 32, kernel_size=(channels, 1), stride=(1, 1), groups=16),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8)),
            nn.Dropout(0.5)
        )

        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, channels, timepoints)
            dummy_output = self.conv_block(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)

        self.fc_block = nn.Sequential(
            nn.Linear(flattened_size, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.fc_block(x)
        return x


# --- 4. 训练和评估主函数 ---
def main(args):
    """
    主函数，负责执行整个训练和评估流程。
    现在所有参数都从 args 对象中获取。
    """
    # 确定设备
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"使用设备: {device}")

    # a. 加载和分割数据
    dataset = EEGDataset(args.data_path)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # b. 初始化模型、损失函数和优化器
    model = SimpleEEGNet(num_classes=dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("\n模型已初始化，开始训练...")
    print("=" * 60)
    print(f"参数: Epochs={args.epochs}, Batch Size={args.batch_size}, Learning Rate={args.lr}")
    print("=" * 60)

    # c. 训练循环
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # d. 在每个epoch后进行验证
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        print(f"Epoch [{epoch + 1}/{args.epochs}] | "
              f"训练损失 (Train Loss): {avg_train_loss:.4f} | "
              f"验证损失 (Val Loss): {avg_val_loss:.4f} | "
              f"验证准确率 (Val Acc): {accuracy:.4f}")

    print("=" * 60)
    print("训练完成！")


# --- 运行主程序 ---
if __name__ == "__main__":
    # 首先解析命令行参数
    args = parse_args()
    # 将解析后的参数传递给主函数
    main(args)
