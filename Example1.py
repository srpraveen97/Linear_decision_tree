
import numpy as np
import pandas as pd
from DecisionTree_LinearRegression import LinearModelTree as lmt

np.random.seed(42)

col1 = np.linspace(1,150,150)
col2 = []
for x in col1:
    if x <= 50:
        col2.append(2*x + 3 + np.random.normal(loc=0,scale=0.5))
    elif x > 50 and x <= 100:
        col2.append(7 + np.random.normal(loc=0,scale=0.5))
    else:
        col2.append(-3*x + 5 + np.random.normal(loc=0,scale=0.5))
mydata = pd.DataFrame({'col1':col1,'col2':col2})

X_train = mydata.to_numpy()[:,[0]]
y_train = mydata.to_numpy()[:,1]

tree = lmt(reg_features=[0],num_cont=150,max_leaves=3)
model = tree.fit(X_train,y_train)

print(f'The root mean squared error is {model.RMSE(X_train,y_train)}')

