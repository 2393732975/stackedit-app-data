
## Pytorch模型评估+日志记录

步骤：
1. 禁止自动求导
2. 将模型设置为评估模式
```python
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

with torch.no_grad():
	for text, label in iterator:
	outputs = model(text.transpose(0, 1))
	loss = criterion(outputs, label)
	total_loss += loss.item()
	_, predicted = torch.max(outputs.data, 1)
	
	accuracy = accuracy_score(label,predicted )
	precision = precision_score(label,predicted )
	recall = recall_score(label,predicted )
	f1 = f1_score(label,predicted )
	
return total_loss / len(iterator), accuracy, precision, recall, f1
```


<!--stackedit_data:
eyJoaXN0b3J5IjpbNTM0NDU2NTk1LDE0MDQwMTg3NzVdfQ==
-->