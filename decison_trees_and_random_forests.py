'''
---------------------------------------Decision trees & Random Forsets Project------------------------------------------
[-] A practice project to understand Decison trees and Random Forests
[-] Dataset taken from LendingClub.com
[-] REQUIREMENTS: Python3 with libraries numpy, pandas, seaborn, matplotlib, scikit-learn
--* Install the requirements using the following pip command without the quotes and execute this command in linux terminal or cmd or powershell from the location where 'req.txt' file is present *--
Linux: "sudo pip install -r req.txt"
CMD & Windows PowerShell: "pip install -r req.txt"
------------------------------------------------------------------------------------------------------------------------
'''

# Problem: Lending Club connects people who need money (borrowers) with people who have money (investors). Hopefully, as an investor you would want to invest in people who showed a profile of having a high probability of paying you back. Create a model that will help predict to this.

# import data pre-processing libraries
import numpy as np
import pandas as pd
# import data visualistaion libraries
import seaborn as sns
import matplotlib.pyplot as plt
# import scikit-learn library to use machine learning models
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# read the loan_data.csv file using pandas
loans = pd.read_csv('loan_data.csv')

# analyse the structure of the dataset
print(loans.info())
print(loans.describe())
print(loans.head())

# set style for the graph
sns.set_style("whitegrid")

# create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome
plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
plt.show()

# create a similar figure, except this time select by the not.fully.paid column
plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')
plt.show()

# create a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid
sns.countplot(x = 'purpose', hue = 'not.fully.paid', data = loans)
plt.show()

# create a jointplot and see the trend between FICO score and interest rate
sns.jointplot(x = 'fico', y='int.rate', data = loans)
plt.show()

# create lmplots to see if the trend differed between not.fully.paid and credit.policy
plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate', x='fico', data=loans, hue='credit.policy', col='not.fully.paid', palette='Set1')
plt.show()

# create a list of 1 element containing the string 'purpose'
cat_feats = ['purpose']
# use pd.get_dummies(loans,columns=cat_feats,drop_first=True) to create a fixed larger dataframe that has new feature columns with dummy variables
final_data = pd.get_dummies(loans, columns = cat_feats, drop_first = True)
# analyse the new dataframe 
print(final_data.info())

# create train and test variables for X and y where test size will be 30% and the remaining 70% will be used to train our model
X_train, X_test, y_train, y_test = train_test_split(final_data.drop('not.fully.paid' ,axis = 1), final_data['not.fully.paid'], test_size = 0.3, random_state = 101)

# DECISION TREES
# create an instance for DecisionTreeClassifier
dtree = DecisionTreeClassifier()
# fit/train the model
dtree.fit(X_train, y_train)
# predict using the test variable of features
predictions = dtree.predict(X_test)

# evaluate by analysing the classification report and confusion amtrix
# classification report
print(classification_report(y_test, predictions))
# confusion matrix
print(confusion_matrix(y_test, predictions))

# RANDOM FORESTS
# create an instance for RandomForestClassifier with 200 estimators
rfc = RandomForestClassifier(n_estimators = 200)
# fit/train the model
rfc.fit(X_train, y_train)
# predict using the test variable of features
rfc_pred = rfc.predict(X_test)

# evaluate by analysing the classification report and confusion amtrix
# classification report
print(classification_report(y_test, rfc_pred))
# confusion matrix
print(confusion_matrix(y_test, rfc_pred))

# neither of the models did well, more feature engineering is required
print("\nNeither of the models did well for this dataset. More features are required.\nDegree of Feature Engineering should be increased.")