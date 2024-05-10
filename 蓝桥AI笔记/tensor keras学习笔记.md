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
>第三：sgd优化器得换成其他的，如adam，或者改变学习率

## 交叉熵损失函数
如果输出神经元是线性的，那么二次代价函数（均方误差：mse）就是一种合适的选择。如果输出神经元是S型函数，那么比较适合用交叉熵代价函数。

分类：

>loss='binary_crossentropy'

二分类交叉熵损失函数适用于二分类问题，其中目标标签只有两个类别。这种情况下，输出层通常使用 sigmoid 激活函数，输出范围在 0 到 1 之间，代表了某一类别的概率。二分类交叉熵损失函数通常更适合于处理这种类型的输出。

>loss='categorical_crossentropy'

多分类交叉熵损失函数适用于多分类问题，其中目标标签有多个类别。这种情况下，输出层通常使用 softmax 激活函数，输出向量的每个元素表示每个类别的概率。多分类交叉熵损失函数能够更好地处理多类别输出的情况，并且与 softmax 激活函数配合使用，更适合于训练多分类问题的神经网络模型。


## 拟合
![输入图片说明](/imgs/2024-05-10/ICDPbjCoAapCR0iQ.png)

![输入图片说明](/imgs/2024-05-10/U5qD7QP5MChnU6gt.png)

解决b'n
<!--stackedit_data:
eyJoaXN0b3J5IjpbNjU1Njg1OTQyLDE0MjQ1NTIzNjMsMTM5Mz
UzMTY3NCw1ODUyNjE2MDAsLTI5ODY2NjQ4NV19
-->