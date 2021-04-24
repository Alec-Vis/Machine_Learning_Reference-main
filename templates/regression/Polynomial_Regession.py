"""
Created on Feb 21 2021

@author: Alec Vis

Polynomial Linear Regression Template

scenario for Data set and analysis:
    we are in an HR department and looking to hire a new employee
    he claims that he was paid $160,000 at his old job and that he want at least that much at this job
    We want to determine if he is lying
    Data set represents data pulled from a website such as glassdoor

We will not be splitting the data set because we already have the test value that we want.
    So we want to produce the most accurate model we can using the entire data set.

Each title has a level based on the title of position
    our candidate in question has 2 years of experience
    therefore we will have his
"""

# Importing The Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import Data
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:-1].values
Y = dataset.iloc[:,-1].values

# linear model fit
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

# train Polynomial model
from sklearn.preprocessing import PolynomialFeatures

    # input for PolynomialFeatures is the power of the polynomial equation
poly_reg = PolynomialFeatures(degree = 4)
    # create new matrix of features based on position levels and squares of position levels
    # this will take the independent variable and generate a numSample by DegreeOfPolynomial sized array
    # would this accept a independent variable matrix? What would occur in this case?
        # YES! this function fits a polynomial to a linear matrix, the case of 2 variables [x1, x2], it would output
        # [1, x1, x2, x1**2, x1*x2, x2**2]
X_poly = poly_reg.fit_transform(X)

# Polynomial model fit
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

# Visualizing the linear Model
    # Note: '@' is matrix multiplication
    # The predict method is a matrix multiplication between the model and the input values
    # Therefore their internal dimensions need to match
plt.scatter(X, Y, color='red')
plt.plot(X,lin_reg.predict(X), color='blue')
plt.title('did he lie? (Linear)')
plt.xlabel('Position level')
plt.ylabel('Salary in gold sheckles')
plt.show()

# Visualizing the polynomial regression model
plt.scatter(X, Y, color='red')
plt.plot(X,lin_reg_2.predict(X_poly), color='blue')
plt.title('did he lie? (polynomial)')
plt.xlabel('Position level')
plt.ylabel('Salary in gold sheckles')
plt.show()

    # this fourth degree polynomial model is over fitted model
    # However this is OK because of the Very special case for what we are after

print(lin_reg.coef_)
print(lin_reg.intercept_)
print(lin_reg.predict([[6.5]])) # returns ~$330,000 ; this is clearly wrong and justifies a polynomial model

print(lin_reg_2.coef_)
print(lin_reg_2.intercept_)
# Because the internal internal dimensions need to match for the matrix multiplication we use the poly_reg.fit_transform
    #method. this will take the X input value and create the polynomial equation
    # ex with degree = 2
    # [[2]] ---transform---> [[1,2,4]]
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))) # ~$158,000; therefore he was telling the truth