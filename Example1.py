
import numpy as np
import pandas as pd
from DecisionTree_LinearRegression import LinearModelTree as lmt

col1 = np.linspace(1,150,150)
col2 = []
for x in col1:
    if x <= 50:
        col2.append(0.3*x + 6 + np.random.normal(loc=0,scale=0.5))
    elif x > 50 and x <= 100:
        col2.append(7 + np.random.normal(loc=0,scale=0.5))
    else:
        col2.append(-0.1*x + 5 + np.random.normal(loc=0,scale=0.5))
mydata = pd.DataFrame({'col1':col1,'col2':col2})

co1 = [20,43,67,89,121,139]
co2 = []
for x in co1:
    if x <= 50:
        co2.append(0.3*x + 6)
    elif x > 50 and x <= 100:
        co2.append(7)
    else:
        co2.append(-0.1*x + 5) 
mydata1 = pd.DataFrame({'co1':co1,'co2':co2})

X_train = mydata.to_numpy()[:,[0]]
y_train = mydata.to_numpy()[:,1]

X_test = mydata1.to_numpy()[:,[0]]
y_test = mydata1.to_numpy()[:,1]


tree = lmt(reg_features=[0])
model = tree.fit(X_train,y_train)

print(model.predict(X_test))
print(f'The root mean squared error is {model.RMSE(X_test,y_test)}')