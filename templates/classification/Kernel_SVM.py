""" Kernel Support Vector Machine:
This is for a non-linearly separable dataset
This will map the dataset to a higher dimension so that we can separate the data in a linear way
    think mapping a 1 dim dataset into 2 dim to create a line through it
    
The draw back with this method is that it is computationally intensive mapping to a higher dimension
    So to achieve a similar result as this without going into a higher dimensional space we will use a technique
    called the 'Kernel Trick'
A kernel function creates a decision boundary without mapping points into a higher dimensional space
    thereby avoiding the computational difficulty of going into a higher dimension
"""



# Import Modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Import Data
dataset = pd.read_csv('Social_Network_Ads.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:, -1].values


# Split the data into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Train K-NN model
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train, y_train)


# test a prediction on test values
# age = 30
# salary = 87000
# print(classifier.predict(sc.transform([[age, salary]]))) # need to scale to match feature scaling

# predict the test set results
y_pred = classifier.predict(x_test)
# np.set_printoptions(precision=2) # do not need this because we are only dealing with integers
# Compare real results to predicted results
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))


# Making a confusion Matrix, this will show up exactly how many correct and incorrect predictions
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
# Measure accuracy of model
acc_scr = accuracy_score(y_test, y_pred)
print(acc_scr)