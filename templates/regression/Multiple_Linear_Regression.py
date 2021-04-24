"""
Created on Feb 21 2021

@author: Alec Vis

Multiple Linear Regression Template

Assumptions:
    Linearity
    Homoscedasticity
    Multivariate normality
    Independence of errors
    Lack of Multicollinearity

Methods to build Model:
    All-In - include all variables, performed when:
        >prior knowledge and you know all are significant
        >you need to include all variable
        >preparing for backward elimination
    Backward Elimination - start with all variables and remove them until only significant variables remain (fastest Model)
        1. select significance level
        2. fit the full model
        3. consider the predictor with highest p-value
        4. remove predictor
        5. Fit model without this variable
        6. repeat steps 3-5 until the variable with highest pvalue is still  less than sigfig level
    Forward Selection - add significant variables one at a time
        1. set sigfig level (SL)
        2. fit all simple regression models y ~ Xn and select the variable with the lowest P-value one
        3. keep variable and all possible second variables
        4. Consider the predictor with the lowest p-value. if P<SL and repeat 3 until P<SL is not true
    BiDirectional - both forward and backward
        1. set sigfig level (SL)
        2. perform next step on forward selection (new variable must satifisfy P<SL)
        3. perform ALL steps on backwards elimination (old Variables must have P<SL)
        4. repeat step 2-3 until no variables can enter or exit
    ALL Possible Models - Very resource intensive
        1. set criterion
        2. construct all possible models
        3. select the one with the best criterion
"""

# Importing The Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Data
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,-1].values


# Taking Care of missing Data
'''from.impute imp sklearnort SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3])'''

# Enconding Categorical Data - Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer (transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

#print(X)

# Encoding Categorical Data - Dependent Variable
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

# Training the Multiple Linear Regression Model on Training Set
    # Note the classes in sklearn will take care of avoiding the Dummy Variable trap and the best elimination technique
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the Test set Results
Y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
    # needed to reshape the Y_pred/test vectors into columns and concatenate horizonally
print(np.concatenate((Y_pred.reshape(len(Y_pred), 1), Y_test.reshape(len(Y_test), 1)), axis=1))

# Prediction for new result
print(regressor.predict([[1, 0, 0, 160000, 130000, 300000]]))

# Equation of Model
print(regressor.coef_)
print(regressor.intercept_)
