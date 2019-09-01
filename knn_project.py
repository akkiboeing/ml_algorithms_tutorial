'''
---------------------------------------K Nearest Neighbors Classification Project---------------------------------------
[-] A practice project to understand KNN and data analytics
[-] REQUIREMENTS: Python3 with libraries numpy, pandas, seaborn, matplotlib, scikit-learn
--* Install the requirements using the following pip command without the quotes and execute this command in linux terminal or cmd or powershell from the location where 'req.txt' file is present *--
Linux: "sudo pip install -r req.txt"
CMD & Windows PowerShell: "pip install -r req.txt"
------------------------------------------------------------------------------------------------------------------------
'''

# Problem: Given a dataset with anonymous column values, predict the TARGET CLASS based on the anonymous cloumns as features using KNN

# import data pre-processing libraries
import numpy as np
import pandas as pd
# import data visualistaion libraries
import matplotlib.pyplot as plt
import seaborn as sns
# import scikit-learn library to use machine learning models
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# read the KNN_Project_Data csv file using pandas
df = pd.read_csv("KNN_Project_Data", index_col = 0)

# analyse the structure of the dataset
print(df.head())

# set style for the graph
sns.set_style("whitegrid")

# do a pairplot for the dataset with hue as 'TARGET CLASS' and analyse the relations between the attributes
sns.pairplot(df, hue="TARGET CLASS" , palette="coolwarm", diag_kind = 'hist')
plt.show()

# create an instance for StandardScaler 
scaler = StandardScaler()

# fit the dataframe without the TRAGET CLASS (since we need to predict it) and transform it to get standardised values  
scaler.fit(df.drop('TARGET CLASS', axis = 1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis = 1))

# create a new dataframe using scaled_features
df_feat = pd.DataFrame(scaled_features, columns = df.columns[:-1])
# analyse the scaled dataframe
print(df_feat.head())

# X denotes the features based on which the prediction should be done
X = scaled_features
# y denotes the history of data which would used to do the prediction 
y = df['TARGET CLASS']

# create train and test variables for X and y where test size will be 30% and the remaining 70% will be used to train our model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

# create an instance for KNeighbortsClassfier with just 1 neighbor(s)
kn = KNeighborsClassifier(n_neighbors = 1)
# fit/train the model
kn.fit(X_train, y_train)
# predict the next y values by passing the test variable of features using the model's predict method
predictions = kn.predict(X_test)

# Evaluavtion: print the confusion matrix & classification report for knowing accuracy and error rate
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# minimizing the error rate
# create an empty list anmed error_rate
error_rate = []

# fit/train the KNN model for nieghbors in range 1 to 40 and append the average error to error_rate list
for ctr in range(1,40):
    knn = KNeighborsClassifier(n_neighbors = ctr)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))

# plot the error_rate for each neighbors in the KNN and analyse it to get the value of neighbors for which error rate is minimum
plt.figure(figsize = (10,8))
plt.plot(range(1,40), error_rate, color = 'blue', linestyle = 'dashed', marker = 'o', markersize = 10, markerfacecolor = 'red')
plt.title('Error Rate vs K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()    

# minimum error rate analysed from the previous visualisation is n_neighbors=37
# create an instance for KNeighbortsClassfier with just 37 neighbors
knn = KNeighborsClassifier(n_neighbors = 37)
# fit/train the model
knn.fit(X_train, y_train)
# predict the next y values by passing the test variable of features using the model's predict method
prediction = knn.predict(X_test)
# Evaluavtion: print the confusion matrix & classification report for knowing accuracy and error rate
print("WITH K=37\n\n{}\n\n{}".format(confusion_matrix(y_test, prediction),classification_report(y_test, prediction)))