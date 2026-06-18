# 深度学习基础与 Transformer 原理讲义（课程课件摘要）

## 1. 神经网络训练核心概念
- 前向传播：输入经各层计算得到输出。
- 损失函数：衡量预测与真实标签差距，分类常用交叉熵，回归常用 MSE。
- 反向传播：链式法则计算各参数梯度。
- 优化器：SGD、Adam 等，按梯度更新参数。Adam 自适应学习率，收敛快，常用默认 lr=1e-3。
- 学习率：最重要超参数之一，影响收敛速度与稳定性。

## 2. 过拟合与正则化
- 过拟合表现：训练 loss 持续下降但验证 loss 上升。
- 应对：Dropout、权重衰减（weight_decay）、数据增强、早停（early stopping）。

## 3. Batch Normalization
- 作用：归一化每层输入，加速收敛、缓解梯度消失。
- 注意：训练用 model.train()，推理必须切 model.eval()，否则 BN 统计量错误。

## 4. Transformer 原理
- 核心是自注意力机制（Self-Attention）：Q、K、V 三个矩阵，Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V。
- 多头注意力（Multi-Head）：并行多组注意力捕捉不同子空间信息。
- 位置编码：Transformer 无循环结构，需加位置编码注入序列顺序信息。
- 编码器-解码器结构：编码器堆叠自注意力+前馈网络，解码器额外含交叉注意力。
- 残差连接 + LayerNorm：每个子层后接 Add & Norm，稳定深层训练。

## 5. 常见调参经验
- 先用小数据/小模型快速验证流程跑通，再放大规模。
- 学习率预热（warmup）+ 衰减常用于 Transformer 训练。
- batch_size 与学习率通常同向调整（大 batch 配大 lr）。
