
## Pytorch模型评估+日志记录

步骤：
1. 禁止自动求导
2. 将模型设置为pin'gu
```python
# 禁用自动求导
with torch.no_grad():
    # 将模型设置为评估模式
    model.eval()

    # 使用模型对数据进行预测
    outputs = model(inputs)

    # 计算损失
    loss = criterion(outputs, labels)

    # 计算准确率
    accuracy = torch.nn.functional.accuracy(outputs, labels)

    # 计算精度、召回率和 F1 值
    precision = sklearn.metrics.precision_score(labels, outputs)
    recall = sklearn.metrics.recall_score(labels, outputs)
    f1 = sklearn.metrics.f1_score(labels, outputs)
    # 输出指标值
    print("Loss:", loss.item())
    print("Accuracy:", accuracy.item())
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
```


<!--stackedit_data:
eyJoaXN0b3J5IjpbLTMwMTY5MzU1Ml19
-->