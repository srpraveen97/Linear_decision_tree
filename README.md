# Decision trees with multiple linear regression models

The file DecisionTree_LinearRegression.py contains the class LinearModelTree which is used to perform a decision tree algorithm with a linear regression model at the leaf nodes. 

The purpose of this algorithm is to use a regular decision tree algorithm to build the tree and to use a linear regression model in each of the leaf nodes. A common example for this is the electricity demand forecasting. There is a piecewise linear relationship between temperature and electricity demand, depending on the seasons, hour of the day, etcetera. An example on how to use this algorithm is shown below.'

### When to use this algorithm?

In practice, we do not generally know apriori if the data is piecewise linear and even if we did, it is hard to determine the pivot points. If we decide the data to be piecewise linear either based on visualization or domain knowledge, we could use the linear model tree algorithm.

### Example-1

I am going to illustrate the use of the algorithm with a single variable toy example. Of course we could always fit three simple linear regression models, but linear model tree algorithm is used for illustration. Consider a piecewise linear equation as shown below:

<a href="https://www.codecogs.com/eqnedit.php?latex=y&space;=&space;\left\{&space;\begin{array}{ll}&space;2x&space;&plus;&space;3&space;&&space;x\leq&space;50&space;\\&space;7&space;&&space;50&space;<&space;x\leq&space;100&space;\\&space;-3x&space;&plus;&space;5&space;&&space;100&space;<&space;x\leq&space;150&space;\\&space;\end{array}&space;\right.\" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y&space;=&space;\left\{&space;\begin{array}{ll}&space;2x&space;&plus;&space;3&space;&&space;x\leq&space;50&space;\\&space;7&space;&&space;50&space;<&space;x\leq&space;100&space;\\&space;-3x&space;&plus;&space;5&space;&&space;100&space;<&space;x\leq&space;150&space;\\&space;\end{array}&space;\right.\" title="y = \left\{ \begin{array}{ll} 2x + 3 & x\leq 50 \\ 7 & 50 < x\leq 100 \\ -3x + 5 & 100 < x\leq 150 \\ \end{array} \right.\" /></a>

In Example1.py, I generated toy data (train and test) for the above function with random noise.

```python
from DecisionTree_LinearRegression import LinearModelTree as lmt
```

```python
tree = lmt(reg_features=[0])
model = tree.fit(X_train,y_train)
```

```python
model.predict(X_test)
model.RMSE(X_test,y_test)
```

