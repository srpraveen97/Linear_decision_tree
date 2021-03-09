
import numpy as np
import pandas as pd
from DecisionTree_LinearRegression import LinearModelTree as lmt

np.random.seed(42)

X_train = np.linspace(-2,2,1000).reshape(-1,1)
y_train = np.array([x*(x-1)*(x-2)*(x+2) + x*np.random.normal(loc=0,scale=0.5) for x in X_train])

tree = lmt(reg_features=[0],max_depth=2)
model = tree.fit(X_train,y_train)

print(f'The root mean squared error is {model.RMSE(X_train,y_train)}')