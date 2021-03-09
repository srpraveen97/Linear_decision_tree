
import numpy as np
import pandas as pd
from DecisionTree_LinearRegression import LinearModelTree as lmt

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

mpl.rcParams['figure.figsize'] = (18,12)
mpl.rcParams['axes.grid'] = False
from matplotlib.axes._axes import _log as matplotlib_axes_logger
matplotlib_axes_logger.setLevel('ERROR')
sns.set()
sns.set_style('whitegrid')

np.random.seed(42)

X_train = np.linspace(-2,2,500).reshape(-1,1)
y_train = np.array([x*(x-1)*(x-2)*(x+2) + np.random.normal(loc=0,scale=0.1)  for x in X_train])

plt.scatter(X_train,y_train)

tree = lmt(reg_features=[0],max_depth=1)
model = tree.fit(X_train,y_train)

tree1 = lmt(reg_features=[0],max_depth=2)
model1 = tree1.fit(X_train,y_train)

model1.tree_param(model1.final_tree)

tree2 = lmt(reg_features=[0],max_depth=3)
model2 = tree2.fit(X_train,y_train)

model2.tree_param(model2.final_tree)

model2.tree_param(model2.final_tree)

tree3 = lmt(reg_features=[0],max_depth=4)
model3 = tree3.fit(X_train,y_train)





plt.show()

print(f'The root mean squared error is {model.RMSE(X_train,y_train)}')