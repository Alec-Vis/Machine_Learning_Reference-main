"""
Created on Feb 21 2021

@author: Alec Vis

Simple Linear Regression Template
"""

# Importing The Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Data
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


# Taking Care of missing Data
'''from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])'''

# Enconding the Independent Variable
'''from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer (transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))'''

# Encoding the Dependent Variable
'''from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)'''

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
X_test[:,3:] = sc.transform(X_test[:,3:])'''

# Training the Simple Linear Regression Model on Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set Results
Y_pred = regressor.predict(X_test)

# Visualizing the Training set results
plt.scatter(X_train,Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(X_test,Y_test, color='red')
    # Note: we do not need to change the X_train predict (will result in same line)
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# To get the Salary of one employee with an arbitrary amount of experience
print('Predicted Salary of an Employee with 12 years of Experience:')
# Note: the Predict Method always expects a 2D Array
# [[num]] --> 2D Array; [num] --> 1D Array; num --> Scalar
print(regressor.predict([[12]]))

# To print the coefficients of the equations
print('X Coefficient:')
print(regressor.coef_)
print('Y intercept:')
print(regressor.intercept_)
