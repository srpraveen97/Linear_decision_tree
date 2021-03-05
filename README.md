# Decision trees with multiple linear regression models

The file DecisionTree_LinearRegression.py contains the class LinearModelTree which is used to perform a decision tree algorithm with a linear regression model at the leaf nodes. 

The purpose of this algorithm is to use a regular decision tree algorithm to build the tree and to use a linear regression model in each of the leaf nodes. A common example for this is the electricity demand forecasting. There is a piecewise linear relationship between temperature and electricity demand, depending on the seasons, hour of the day, etcetera. An example on how to use this algorithm is shown below.

### Example-1

Consider a piecewise linear function as shown below:

<img src="http://www.sciweavers.org/tex2img.php?eq=%5Cbegin%7Bequation%2A%7D%0Ay%20%3D%20%0A%5Cleft%5C%7B%0A%5Cbegin%7Barray%7D%7Bll%7D%0A%20%20%20%20%20%202x%20%2B%203%20%26%20x%5Cleq%2050%20%5C%5C%0A%20%20%20%20%20%207%20%26%2050%20%3C%20x%5Cleq%20100%20%5C%5C%0A%20%20%20%20%20%20-3x%20%2B%205%20%26%20100%20%3C%20x%5Cleq%20150%20%5C%5C%0A%5Cend%7Barray%7D%20%0A%5Cright.%5C%5D%0A%5Cend%7Bequation%2A%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\begin{equation*}y = \left\{\begin{array}{ll}      2x + 3 & x\leq 50 \\      7 & 50 < x\leq 100 \\      -3x + 5 & 100 < x\leq 150 \\\end{array} \right.\]\end{equation*}" width="257" height="68" />

Using a simple linear regression

In progress... ... ...
