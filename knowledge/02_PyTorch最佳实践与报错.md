# PyTorch 官方最佳实践与常见报错

## 1. 张量维度不匹配（shape mismatch）
报错示例：`mat1 and mat2 shapes cannot be multiplied (8x784 and 256x10)`
- 原因：全连接层 nn.Linear(in_features, out_features) 的 in_features 与输入最后一维不符。
- 排查：在 forward 中 print(x.shape)；卷积展平后用 x = x.view(x.size(0), -1)。
- 注意：8 是 batch，784=28*28 是展平后的特征数，应让 Linear 的 in_features=784。

## 2. 设备不匹配（device mismatch）
报错示例：`Expected all tensors to be on the same device`
- 模型和数据要在同一设备：model.to(device)，每个 batch 也要 x = x.to(device)、y = y.to(device)。
- device 统一定义：device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')。

## 3. 数据类型不匹配（dtype mismatch）
报错示例：`expected scalar type Long but found Float`
- CrossEntropyLoss 的 target 必须是 LongTensor（整数类别），用 labels.long()。
- 输入特征一般为 FloatTensor，用 .float()。

## 4. CUDA out of memory（显存溢出）
- 减小 batch_size 是最直接的办法。
- 使用 torch.cuda.amp 混合精度训练。
- 梯度累积：用小 batch 多次 backward 后再 step，等效大 batch。
- 验证/推理阶段用 with torch.no_grad() 关闭梯度，节省显存。
- 及时释放：del 中间变量 + torch.cuda.empty_cache()。

## 5. 标准训练循环模板
```python
model.train()
for epoch in range(epochs):
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
```

## 6. DataLoader 常见问题
- num_workers 在 Windows 下过大可能报错，调试时设为 0。
- 最后一个 batch 不满：drop_last=True 或确保模型可处理变长 batch。
