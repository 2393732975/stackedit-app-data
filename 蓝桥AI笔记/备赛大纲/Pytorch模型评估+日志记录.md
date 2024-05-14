
## Pytorch模型评估+日志记录

步骤：
1. 禁止自动求导
2. 将模型设置为评估模式
```python
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

all_labels = []
all_predictions = []

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

# 写日志文件
with open('training.log', 'w') as log_file:
	
	for epoch in trange(NUM_EPOCHS):
		train_loss = train(model, train_loader, criterion, optimizer)
		val_loss, val_accuracy, precision, recall, f1 = evaluate(model, val_loader, criterion)

		# 按照指定格式记录日志
		log_file.write(f"Epoch: {epoch + 1}\n")
		log_file.write(f"Train Loss: {train_loss:.3f}\n")
		log_file.write(f"Val Loss: {val_loss:.3f} | Val Accuracy: {val_accuracy * 100:.2f}% | Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f}\n")
		log_file.flush() # 确保内容立即写入文件
```


<!--stackedit_data:
eyJoaXN0b3J5IjpbNDM0NTM4NzU3LDg2OTI0MTM2NCwxNDA0MD
E4Nzc1XX0=
-->