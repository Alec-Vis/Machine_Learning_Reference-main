"""
@author Alec Vis, 2/28/2021

Random Forest Regression
step 1:
    pick a random selection of K data points from the data set
step 2:
    Build the decision tree associated to these K data points
step 3:
    choose the number of trees you want to build and repeat steps 1 & 2
step 4:
    given a new data point, have each decision tree try to predict the value of this new point and average the result

This allows the model to be more accurate and stable at the cost of some performance
    typically default values of this are ~500
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
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X, Y)


# Predict single value with Model
print(regressor.predict([[6.5]])) # with more than one feature include them in the 2D array also


# Plot the results, not relavent for larger numbers of features
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
Y_1 = regressor.predict(X_grid)

# plt.figure()
# plt.scatter(X, Y, color="red")
# plt.plot(X_grid, Y_1, color="blue")
# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.show()