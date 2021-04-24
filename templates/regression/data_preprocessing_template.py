# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 18:26:19 2021

@author: Alec Vis

a range in python include lower bound but excludes upper bound

Data PreProcessing Template
"""

# Importing The Libraries
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

##### Importing the dataset #####
dataset = pd.read_csv('Data.csv') 
    # Features, is used to predict Dependent Variable
    # .iloc function locates the indexes [rows, cols]
    # the .values changes the type from dataframe to object
X = dataset.iloc[:,:-1].values
    # Dependent variable, last column 
Y = dataset.iloc[:,-1].values

#print(X)
#print(Y)


##### Taking Care of missing Data (can either remove or add in an average value) #####
from sklearn.impute import SimpleImputer
    # Imputer object which will define how we will replace the missing values
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    # this is determine the appropriate values needed to fill missing cells
imputer.fit(X[:,1:3])
    # this find the empty cells and input the fitted value
X[:,1:3] = imputer.transform(X[:,1:3])

#print(X)

##### Encoding Categorical data: #####
    # we do not want to change the string names of our categories into 1,2,3,...
        # if this is done the machine learning algorithm will interpret an order
        # among these categories, which is a false presumption
    # one hot encoding is used to categorize the data
        # this will turn each category into its own individual column
        # these new columns will be turned into a column of 0s and 1s
    # if there is a category such as Yes/No:
        # then is can appropriote to simply replace the Y/N with 1/0 respectively

# Enconding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
    # syntax transformer=[('type of transformation', NameOfClassThatWillPerformTransformation, [ColumnToBeTransformed])], remainder='WhatWillOccurToOtherColumns')
    # reaminder = 'Passthrough' says that we want to include the remaining columns of the dataset.
#       # if this is not included then we will only recieve the encoded columns
ct = ColumnTransformer (transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X)) #tranform output to numpy array and reasign dataset

#print(X)

# Encoding the Dependent Variable 
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)

#print(Y)

# Splitting the dataset into the Training set Test set
from sklearn.model_selection import train_test_split
    # recommended 80% training data set and 20% test data set
    # last parameter is the pseudorandom number seed
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=1)

#print(X_train)
#print(X_test)
#print(Y_train)
#print(Y_test)


''' Feature Scaling - scaling all variables to prevent one feature from dominating the model
    We want this after splitting because we do not want to include the
    test data in the mean/std deviation of the feature scaling
    test data and training data should be completely seperate from each other
    Thus there will be information leakage on the test set (which should not occur until training is done)
    The test data represents Completely new and instances of the data
    
    Standardization: (x-mean(x))/StdDev(x)
        -3 <= x <= 3 (how scaling affects the data set)
        This will produce meaningful results in all cases
        Therefore, this is the recommended feature scaling
    Normalization: (x-Min(x))/(Max(x)-Min(x))
        -1 <= x <= 1 (how scaling affects the data set)
        recommended only when there is a normal distribution in most of your features
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
    # Note: we will not need to apply feature scaling on the dummy variable
    # because they are already in scaled range
    # scaling will hurt this categorical information
X_train[:,3:] = sc.fit_transform(X_train[:,3:])
    # we want to apply the same transformation to be done on the test data
    # if a different fit method to scale the test data then it reduces our confidence in the results
    # since they are changed differently
X_test[:,3:] = sc.transform(X_test[:,3:])

print(X_train)
print(X_test)

# Evaluating the model performance
# from sklearn.metrics import r2_score
# r2_score(Y_test, Y_pred)
