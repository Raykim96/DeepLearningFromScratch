from sklearn.datasets import load_boston
import pandas as pd
boston = load_boston()
dfx = pd.DataFrame(boston.data, columns = boston.feature_names)
dfy = pd.DataFrame(boston.target, columns =["MEDV"])
boston.target
dir(boston)

df = pd.concat([dfx, dfy], axis=1)

import statsmodels.api as sm
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

bias = 100
X0, y, w = make_regression(
    n_samples=200, n_features=1, bias=bias, noise=10, coef=True, random_state=1
)
X0.shape
X = sm.add_constant(X0)
y = y.reshape(len(y), 1)
w

plt.scatter(X0, y, s=100)
plt.show()