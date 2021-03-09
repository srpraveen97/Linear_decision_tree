
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

col1 = np.linspace(1,150,150)
col2 = []
for x in col1:
    if x <= 50:
        col2.append(0.2*x  + np.random.normal(loc=0,scale=0.5))
    elif x > 50 and x <= 100:
        col2.append(7 + np.random.normal(loc=0,scale=0.3))
    else:
        col2.append(-0.1*x + 15 + np.random.normal(loc=0,scale=0.5))
mydata = pd.DataFrame({'col1':col1,'col2':col2})

X_train = mydata.to_numpy()[:,[0]]
y_train = mydata.to_numpy()[:,1]

plt.scatter(X_train,y_train)


tree = lmt(reg_features=[0],num_cont=150)
model = tree.fit(X_train,y_train)


pred = []
for x in col1:
    if x < 51:
        pred.append(0.1941*x + 0.0386)
    elif x >= 51 and x < 101:
        pred.append(-0.0031*x + 7.2436)
    else:
        pred.append(-0.1012*x + 15.1347)

plt.plot(X_train,pred,color='tab:red')        
plt.show()

print(f'The root mean squared error is {model.RMSE(X_train,y_train)}')

