# -*- coding: utf-8 -*-
"""
说话人识别训练脚本
==================
基于 CNN 的说话人识别（Speaker Recognition）模型训练。
输入为 Mel 频谱图，经卷积特征提取后分类到不同说话人。

支持通过环境变量进行 debug 裁剪（由训练执行工具注入）：
    DEBUG_MODE / MAX_STEPS / BATCH_SIZE / MAX_EPOCHS / DEVICE
"""

import os
import torch
import torch.nn as nn

# ---------------- 运行配置（从环境变量读取，支持 debug 裁剪） ----------------
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
DEBUG = os.getenv("DEBUG_MODE", "0") == "1"
MAX_STEPS = int(os.getenv("MAX_STEPS", "3000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
MAX_EPOCHS = int(os.getenv("MAX_EPOCHS", "1"))

# ---------------- 数据维度 ----------------
N_MELS = 40            # 梅尔频率维
N_FRAMES = 100         # 时间帧
NUM_SPEAKERS = 10      # 说话人类别数
NUM_SAMPLES = 256      # 合成样本数

# ---------------- 超参数 ----------------
LR = 1.0               # 学习率


def make_synthetic_data(n: int):
    """生成合成说话人数据：每个说话人有独特频谱模板（类中心）+ 噪声。"""
    torch.manual_seed(0)
    y = torch.randint(0, NUM_SPEAKERS, (n,))
    centers = torch.randn(NUM_SPEAKERS, 1, N_MELS, N_FRAMES) * 1.0
    x = centers[y] + torch.randn(n, 1, N_MELS, N_FRAMES) * 1.5
    return x, y


class SpeakerNet(nn.Module):
    def __init__(self, n_mels=N_MELS, n_frames=N_FRAMES, num_speakers=NUM_SPEAKERS):
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(16 * n_mels * n_frames, num_speakers)

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def main():
    print(f"[配置] device={DEVICE} debug={DEBUG} max_steps={MAX_STEPS} "
          f"batch_size={BATCH_SIZE} epochs={MAX_EPOCHS} lr={LR}")

    x, y = make_synthetic_data(NUM_SAMPLES)
    x, y = x.to(DEVICE), y.to(DEVICE)

    torch.manual_seed(1)
    model = SpeakerNet().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    step = 0
    n_batches = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE
    for epoch in range(MAX_EPOCHS):
        model.train()
        for b in range(n_batches):
            xb = x[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            yb = y[b * BATCH_SIZE:(b + 1) * BATCH_SIZE]
            if xb.size(0) == 0:
                continue

            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

            step += 1
            if step % 5 == 0 or step == 1:
                acc = (out.argmax(1) == yb).float().mean().item()
                print(f"epoch {epoch} step {step} loss={loss.item():.4f} acc={acc:.3f} lr={LR}")
            if step >= MAX_STEPS:
                break
        if step >= MAX_STEPS:
            break

    print("[完成] 训练结束")


if __name__ == "__main__":
    main()
