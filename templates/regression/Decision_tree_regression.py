"""
@author Alec Vis, 2/27/2021

Decision Tree regression:
    where/how the leaves are split is determined by the the data's information entropy
        i.e. the algorithm will continue splitting until no more information can be added

Do not need Feature scaling for decision tree regression
    the magnitude of the values in the data is not relavent for how the splitting the data is split

This model is not good for this particular dataset because it has only one feature
    Decision tree regression is more suited for high dimensional datasets
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing Data set
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values

# Taking care of missing Data
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputer.fit(X[:, 1:3])
# X[:,1:3] = imputer.transform(X[:,1:3])

# Encoding categorical variables

# Encoding Independent Variable: (assuming order does not matter for indep variable)
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# X = np.array((ct.fit_transform(X)))

# Encoding Dependent Variable: (assuming order matters for dependent variable)
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# Y = le.fit_transform(Y)

# Training the Decision Tree Regression model on the whoel dataset
from sklearn.tree import DecisionTreeRegressor
regr_1 = DecisionTreeRegressor(random_state=0)
regr_2 = DecisionTreeRegressor(max_depth=5)
regr_1.fit(X, Y)
regr_2.fit(X, Y)

# Predict single value with Model
print(regr_1.predict([[6.5]])) # with more than one feature include them in the 2D array also


# Plot the results, not relavent for larger numbers of features
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
Y_1 = regr_1.predict(X_grid)
Y_2 = regr_2.predict(X_grid)
#
# plt.figure()
# plt.scatter(X, Y, color="red")
# plt.plot(X_grid, Y_1, color="blue")
# # plt.plot(X_grid, Y_2, color="yellowgreen")
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.show()