import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Import the data
dataset = pd.read_csv(r'c:/Users/Alec Vis/ML_Templates/data/concrete_data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# summary statistics of data
indep_stats = stats.describe(np.array(x))
dep_stats = stats.describe(np.array(y))

# splitting the training and the test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# train model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# predict values of y
y_pred = regressor.predict(X_test)

# evaluate the model
from sklearn.metrics import r2_score
print(r2_score(y_test, y_pred))

def preprocess_data(data):
    #convert values to float type
    dataset = np.array(data, dtype=np.float)
    # last column is target column, switch 
    X, y = dataset[:,:-1], dataset[:,-1]
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    return X_train, X_train, y_test, y_train


# ============================
# Multilinear regression 
import seaborn as sns
sns.set_theme()

# Plot sepal width as a function of sepal_length across days
g = sns.lmplot(
    data=dataset,
    x="coarse_aggregate", y="concrete_compressive_strength", hue="superplasticizer",
    height=5
)

# Use more informative axis labels than are provided by default
g.set_axis_labels("coarse_aggregate", "compressive Strength")
plt.show()
