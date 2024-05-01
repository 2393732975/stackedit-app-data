# 决策树和K近邻

### 决策树
熵（entropy）： 熵指的是体系的混乱的程度，在不同的学科中也有引申出的更为具体的定义，是各领域十分重要的参量。（大多数学过物理化学的人应该很熟悉这个概念，没什么变化。）

信息论（information theory）中的熵（香农熵）： 是一种信息的度量方式，表示信息的混乱程度，也就是说：信息越有序，信息熵越低。（说白了也是讲信息的混乱程度，我的理解是：一堆信息中不同的类很多，说明混乱程度高，熵就高；一堆信息中只有一两种类，那熵就比较低。）


信息增益（information gain）： 在划分数据集前后信息发生的变化称为信息增益。（就是划分数据到不同子集前后，熵的变化。）

#### sklearn.tree.DecisionTreeClassifier 类
sklearn.tree.DecisionTreeClassifier 类的主要参数为：
-   max_depth 树的最大深度；
-   max_features 搜索最佳分区时的最大特征数（特征很多时，设置这个参数很有必要，因为基于所有特征搜索分区会很「昂贵」）；
-   min_samples_leaf 叶节点的最少样本数。
```
reg_tree = DecisionTreeRegressor(max_depth=5, random_state=17) 
reg_tree.fit(X_train, y_train) 
reg_tree_pred = reg_tree.predict(X_test)
```
### K近邻
最近邻方法（K 近邻或 k-NN）是另一个非常流行的分类方法。当然，也可以用于回归问题。和决策树类似，这是最容易理解的分类方法之一。这一方法遵循紧密性假说：如果样本间的距离能以足够好的方法衡量，那么相似的样本更可能属于同一分类。

k-NN 分类/回归的效果取决于一些参数：

-   邻居数 k。
-   样本之间的距离度量（常见的包括 Hamming，欧几里得，余弦和 Minkowski 距离）。注意，大部分距离要求数据在同一尺度下，例如「薪水」特征的数值在千级，「年龄」特征的数值却在百级，如果直接将他们丢进最近邻模型中，「年龄」特征就会受到比较大的影响。
-   邻居的权重（每个邻居可能贡献不同的权重，例如，样本越远，权重越低）。

#### scikit-learn 的 KNeighborsClassifier 类

`sklearn.neighbors.KNeighborsClassifier` 类的主要参数为：

-   weights：可设为 uniform（所有权重相等），distance（权重和到测试样本的距离成反比），或任何其他用户自定义的函数。
-   algorithm（可选）：可设为 brute、ball_tree、KD_tree、auto。若设为 brute，通过训练集上的网格搜索来计算每个测试样本的最近邻；若设为 ball_tree 或 KD_tree，样本间的距离储存在树中，以加速寻找最近邻；若设为 auto，将基于训练集自动选择合适的寻找最近邻的方法。
-   leaf_size（可选）：若寻找最近邻的算法是 BallTree 或 KDTree，则切换为网格搜索所用的阈值。
-   metric：可设为 minkowski、manhattan、euclidean、chebyshev 或其他。


### 验证模型的质量
-   留置法。保留一小部分数据（一般是 20% 到 40%）作为留置集，在其余数据上训练模型（原数据集的 60%-80%），然后在留置集上验证模型的质量。
-   交叉验证。最常见的情形是 k 折交叉验证，如下图所示。
![输入图片说明](/imgs/2024-04-30/0AH8fYoa5r1uiQtZ.png)
在 k 折交叉验证中，模型在原数据集的 𝐾−1K−1 个子集上进行训练（上图白色部分），然后在剩下的 1 个子集上验证表现，重复训练和验证的过程，每次使用不同的子集（上图橙色部分），总共进行 K 次，由此得到 K 个模型质量评估指数，通常用这些评估指数的求和平均数来衡量分类/回归模型的总体质量。

相比留置法，交叉验证能更好地评估模型在新数据上的表现。然而，当你有大量数据时，交叉验证对机器计算能力的要求会变得很高。

### 比较在决策树和最近邻方法（通过实例）

```
# 读入数据
import pandas as pd 
df = pd.read_csv( 'https://labfile.oss.aliyuncs.com/courses/1283/telecom_churn.csv') 
df['International plan'] = pd.factorize(df['International plan'])[0] 
df['Voice mail plan'] = pd.factorize(df['Voice mail plan'])[0] 
df['Churn'] = df['Churn'].astype('int') 
states = df['State'] 
y = df['Churn'] 
df.drop(['State', 'Churn'], axis=1, inplace=True)
```
留置法：
接下来，训练 2 个模型：决策树和 k-NN。一开始，我们并不知道如何设置模型参数能使模型表现好，所以可以使用随机参数方法，假定树深（max_dept）为 5，近邻数量（n_neighbors）为 10。
```
from sklearn.model_selection 
import train_test_split, StratifiedKFold from sklearn.neighbors 
import KNeighborsClassifier from sklearn.tree 
import DecisionTreeClassifier 
X_train, X_holdout, y_train, y_holdout = train_test_split(df.values, y, test_size=0.3, random_state=17) 
tree = DecisionTreeClassifier(max_depth=5, random_state=17) 
# 创建了一个最大深度为5的决策树分类器对象`tree`，并设置了随机种子为17。
knn = KNeighborsClassifier(n_neighbors=10) tree.fit(X_train, y_train) knn.fit(X_train, y_train)
# 创建了一个K近邻分类器对象`knn`，设置了邻居数量为10。
```
使用准确率（Accuracy）在留置集上评价模型预测的质量。
```
from sklearn.metrics import accuracy_score 

tree_pred = tree.predict(X_holdout) 
accuracy_score(y_holdout, tree_pred)

knn_pred = knn.predict(X_holdout) 
accuracy_score(y_holdout, knn_pred)
```
从上可知，决策树的准确率约为 94%，k-NN 的准确率约为 88%，于是仅使用我们假定的随机参数（即没有调参），决策树的表现更好。


现在，使用交叉验证确定树的参数，对每次分割的 max_dept（最大深度 h）和 max_features（最大特征数）进行调优。`GridSearchCV()` 函数可以非常简单的实现交叉验证，下面程序对每一对 max_depth 和 max_features 的值使用 5 折验证计算模型的表现，接着选择参数的最佳组合。
```
from sklearn.model_selection import GridSearchCV, cross_val_score 
tree_params = {'max_depth': range(5, 7), 'max_features': range(16, 18)} 
tree_grid = GridSearchCV(tree, tree_params, cv=5, n_jobs=-1, verbose=True) 
# 创建了一个GridSearchCV对象`tree_grid`，用于在参数网格上进行交叉验证网格搜索。参数`tree`是要搜索的模型对象，`tree_params`是参数网格，`cv=5`表示使用5折交叉验证，`n_jobs=-1`表示使用所有可用的CPU核心进行计算，`verbose=True`表示输出详细的信息。
tree_grid.fit(X_train, y_train)
```

现在，再次使用交叉验证对 k-NN 的 k 值（即邻居数）进行调优。
```
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
knn_pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier(n_jobs=-1))]) 
knn_params = {'knn__n_neighbors': range(6, 8)} 
knn_grid = GridSearchCV(knn_pipe, knn_params, cv=5, n_jobs=-1, verbose=True) 
knn_grid.fit(X_train, y_train) 
knn_grid.best_params_, knn_grid.best_score_
```

#### 决策树

优势：

-   生成容易理解的分类规则，这一属性称为模型的可解释性。例如它生成的规则可能是「如果年龄不满 25 岁，并对摩托车感兴趣，那么就拒绝发放贷款」。
-   很容易可视化，即模型本身（树）和特定测试对象的预测（穿过树的路径）可以「被解释」。
-   训练和预测的速度快。
-   较少的参数数目。
-   支持数值和类别特征。

劣势：

-   决策树对输入数据中的噪声非常敏感，这削弱了模型的可解释性。
-   决策树构建的边界有其局限性：它由垂直于其中一个坐标轴的超平面组成，在实践中比其他方法的效果要差。
-   我们需要通过剪枝、设定叶节点的最小样本数、设定树的最大深度等方法避免过拟合。
-   不稳定性，数据的细微变动都会显著改变决策树。这一问题可通过决策树集成方法来处理（以后的实验会介绍）。
-   搜索最佳决策树是一个「NP 完全」（NP-Complete）问题。了解什么是 NP-Complete 请点击 [_这里_](https://baike.baidu.com/item/NP-Complete/15961931?fr=aladdin)。实践中使用的一些推断方法，比如基于最大信息增益进行贪婪搜索，并不能保证找到全局最优决策树。
-   倘若数据中出现缺失值，将难以创建决策树模型。Friedman 的 CART 算法中大约 50% 的代码是为了处理数据中的缺失值（现在 sklearn 实现了这一算法的改进版本）。
-   这一模型只能内插，不能外推（随机森林和树提升方法也是如此）。也就是说，倘若你预测的对象在训练集所设置的特征空间之外，那么决策树就只能做出常数预测。比如，在我们的黄球和蓝球的例子中，这意味着模型将对所有位于 >19 或 <0 的球做出同样的预测。

#### 最近邻方法

优势：

-   实现简单。
-   研究很充分。
-   通常而言，在分类、回归、推荐问题中第一个值得尝试的方法就是最近邻方法。
-   通过选择恰当的衡量标准或核，它可以适应某一特定问题。

劣势：

-   和其他复合算法相比，这一方法速度较快。但是，现实生活中，用于分类的邻居数目通常较大（100-150），在这一情形下，k-NN 不如决策树快。
-   如果数据集有很多变量，很难找到合适的权重，也很难判定哪些特征对分类/回归不重要。
-   依赖于对象之间的距离度量，默认选项欧几里得距离常常是不合理的。你可以通过网格搜索参数得到良好的解，但在大型数据集上的耗时很长。
-   没有理论来指导我们如何选择邻居数，故而只能进行网格搜索（尽管基本上所有的模型，在对其超参数进行调整时都使用网格搜索的方法）。在邻居数较小的情形下，该方法对离散值很敏感，也就是说，有过拟合的倾向。
-   由于「维度的诅咒」，当数据集存在很多特征时它的表现不佳。
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTE4NDQ4MDY1XX0=
-->