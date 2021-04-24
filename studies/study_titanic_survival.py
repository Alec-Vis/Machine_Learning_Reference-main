"""Study to determine which parameters were most influenctial in the survival of passengers on the titanic
"""


import pandas as pd
import numpy as np
data = pd.read_csv(r'..\data\titanic.csv')

data.replace('?', np.nan, inplace=True)
data = data.astype({"age": np.float64, "fare": np.float64})

import seaborn as sns
import matplotlib.pyplot as plt

#visualize titanic data
fig, axs = plt.subplots(ncols=5, figsize=(30,5))
sns.violinplot(x="survived", y="age", hue="sex", data=data, ax=axs[0])
sns.pointplot(x="sibsp", y="survived", hue="sex", data=data, ax=axs[1])
sns.pointplot(x="parch", y="survived", hue="sex", data=data, ax=axs[2])
sns.pointplot(x="pclass", y="survived", hue="sex", data=data, ax=axs[3])
sns.violinplot(x="survived", y="fare", hue="sex", data=data, ax=axs[4])
plt.show()

# replace male and female strings to integer to run a correlation on the data
data.replace({'male':1, 'female':0}, inplace=True)
# Run Correlations
data.corr().abs()[["survived"]]
''' it looks like being along matters a lot but the number of parents/siblings didn't
therefore we will change the data set to look at whether or not having siblings 
influenced survivability
'''

# combine sibling and parent columns
data['relatives'] = data.apply(lambda row: int((row['sibsp'] + row['parch']) > 0), axis=1)
# new correlation
data.corr().abs()[['survived']]
''' This new correlation supports our hypothesis that only the fact that you had siblings mattered
not how many of them you had
'''

# Create new data set and drop unnecessary data
data = data[['sex', 'pclass', 'age', 'relatives', 'fare', 'survived']].dropna()

# split the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data[['sex', 'pclass', 'age', 'relatives', 'fare']], data.survived, test_size=0.2)

# normalize the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

# training a ML Model
''' we are modeling a classification problem (Survived or NOT)
Gaussian Naive Bayes:
> assumed that features are assumed to be gaussian
> assuming conditional independence between every pair of features given the variable class
'''
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train,y_train)

# Testing Model
from sklearn import metrics
predict_test = model.predict(x_test)
print(metrics.accuracy_score(y_test, predict_test))

'''
# ====================================
# Let's make a neural network
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# import neural network models
# Sequential which is a layered neural network wherein there are mutliple layers feed into each other in sequence
from keras.models import Sequential
from keras.layers import Dense

model = Sequential()
# 5 inputs, therefore this first layer will be dim 5
model.add(Dense(5, kernel_initializer='uniform', activation='relu', input_dim = 5))
# middle layer will be kept at 5 for simplicity, don't see a reason to change this
model.add(Dense(5,kernel_initializer='uniform',activation='relu'))
# last layer si one dimensional because all we want to know is whether or not is a yes/no survival
model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
'''