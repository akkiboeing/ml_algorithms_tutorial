'''
---------------------------------------Advertisement Logistic Regression Project---------------------------------------
[-] A practice project to understand Logistic Regression and data analytics
[-] REQUIREMENTS: Python3 with libraries numpy, pandas, seaborn, matplotlib, scikit-learn
--* Install the requirements using the following pip command without the quotes and execute this command in linux terminal or cmd or powershell from the location where 'req.txt' file is present *--
Linux: "sudo pip install -r req.txt"
CMD & Windows PowerShell: "pip install -r req.txt"
-----------------------------------------------------------------------------------------------------------------
'''

# Problem: Given a fake advertising dataset predict whether or not a particular user clicks on an Advertisement

# import data pre-processing libraries
import numpy as np
import pandas as pd
# import data visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
# import scikit-learn library to use machine learning models
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# read the advertisement.csv file (our dataset) using pandas
ad_data = pd.read_csv("advertising.csv")

# analyse the structure of the dataset 
print(ad_data.head())
print(ad_data.info())
print(ad_data.describe())

# set style for the graph
sns.set_style("whitegrid")

# plot a histogram chart for column 'Age'
ad_data['Age'].plot.hist(bins=30)
plt.show()

# plot a jointplot between 'Age' and 'Area Income'
sns.jointplot(x='Age', y='Area Income', data = ad_data)
plt.show()

# plot a jointplot between 'Age' and 'Daily Time Spent on Site'
sns.jointplot(x = 'Age', y = 'Daily Time Spent on Site', data = ad_data, kind = 'kde')
plt.show()

# plot a jointplot between 'Daily Time Spent on Site' and 'Daily Internet Usage'
sns.jointplot(x = 'Daily Time Spent on Site', y = 'Daily Internet Usage', data = ad_data)
plt.show()

# visualise and analyse the dataset by using pairplot
sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr', diag_kind = 'hist')
plt.show()

# X denotes the features based on which the prediction should be done
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
# y denotes the history of data which would used to do the prediction 
y = ad_data['Clicked on Ad']

# create train and test variables for X and y where test size will be 33% and the remaining 67% will be used to train our model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 101)

# create an object/instance for Logistic Regression model
# φ(z) = 1/(1+e^(-z)) {Sigmoid or Logisitic function}
logm = LogisticRegression()

# train/fit the data using the Logistic Regression object 
logm.fit(X_train, y_train)

# predict the next y values by passing the test variable of features using the model's predict method
predictions = logm.predict(X_test)

# evaluation 
# print the classification report for our dataset processed by the model
print("Classification Report: \n", classification_report(y_test, predictions))

# print the confusion matrix for calculating accuracy based on type-1 and type-2 errors
conmat = confusion_matrix(y_test, predictions)
print("Confusion matrix: \n", conmat)