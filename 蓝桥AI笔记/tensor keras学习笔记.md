## 线性回归

```
import tensorflow
import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# 生成一个数据
x_data =np.random.rand(100)
noise = np.random.normal(0,0.01,x_data.shape)
y_data = x_data*0.1+0.3 +noise


#显示随机点
#plt.scatter(x_data,y_data)
#plt.show()

# 构建一个顺序模型
model = Sequential()

#在模型中添加全连接层
model.add(Dense(units=1,input_dim=1)) 

#sgd:随机梯度下降法 ——————优化器
#mse:均方误差 ————损失函数
model.compile(optimizer='sgd',loss='mse')

#训练
for step in range(3001):
	cost = model.train_on_batch(x_data,y_data)
	if step%500==0：
		print('cost:',cost)
		
#打印权值和偏置值
W,b=model.layers[0].get_weights()
print('W:',W,'b:',b)

y_pre= model.predict(x_data)  
plt.scatter(x_data,y_data)
plt.plot(x_data, y_pre, 'r-', lw=3)
plt.show()
```

## 非线性回归
与线性回归不同的是：
>第一：不能只加一层Dense
```
# 除了第一层必须有输入，后面的可以无输入，因为默认是前一层的units（输出）
model.add(Dense(units=10,input_dim=1)) 
model.add(Dense(units=1)) 
```
>第二：激活函数
```
# 除了第一层必须有输入，后面的可以无输入，因为默认是前一层的units（输出）
model.add(Dense(units=10,input_dim=1)) 
model.add(Dense(units=1)) 
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbNTg1MjYxNjAwLC0yOTg2NjY0ODVdfQ==
-->