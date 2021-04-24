"""Support Vector Machine:
This algorithm maximizes the distance of points from the regression line that splits the data
    rather than looking at the most stock standard derivatives of a class, it looks at the borderline cases to create the model
"""



# Import Modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Import Data
dataset = pd.read_csv(r'Social_Network_Ads.csv')
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
classifier = SVC(kernel='linear', random_state=0)
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
