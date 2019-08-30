'''
---------------------------------------Ecommerce Linear Regression Project---------------------------------------
[-] A practice project to understand Linear Regression and data analytics
[-] REQUIREMENTS: Python3 with libraries numpy, pandas, seaborn, matplotlib, scikit-learn
--* Install the requirements using the following pip command without the quotes and execute this command in linux terminal or cmd or powershell from the location where 'req.txt' file is present *--
Linux: "sudo pip install -r req.txt"
CMD & Windows PowerShell: "pip install -r req.txt"
-----------------------------------------------------------------------------------------------------------------
'''

# Problem: A company is trying to decide whether to focus their efforts on their mobile app experience or their website. 

# import data pre-processing libraries
import pandas as pd
import numpy as np
# import data visualization libraries 
import matplotlib.pyplot as plt
import seaborn as sns
# import scikit-learn library to use machiune learning models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# set color palette (styles) for graph
sns.set_palette("GnBu_d")

# read the Ecommerce Customers csv file (our dataset) using pandas
ecom = pd.read_csv("Ecommerce Customers")

# analyse the structure of the dataset 
print(ecom.head())
print(ecom.info())
print(ecom.describe())

# analyse the data present in the dataset and find relational attributes
sns.jointplot(x = 'Time on Website', y = 'Yearly Amount Spent', data = ecom)
sns.jointplot(x = 'Time on App', y = 'Yearly Amount Spent', data = ecom)
sns.jointplot(x = 'Time on App', y = 'Length of Membership', data = ecom, kind = 'hex')
plt.show()

# visualise and analyse the dataset by using pairplot
sns.pairplot(ecom)
plt.show()

# test the linear model analysis for the discovered relational attributes on which Linear regression can be applied
sns.lmplot(x='Length of Membership', y ='Yearly Amount Spent', data = ecom)

# X denotes the features based on which the prediction should be done
X = ecom[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
# y denotes the history of data which would used to do the prediction 
y = ecom['Yearly Amount Spent']

# create train and test variabled for X and y where test siz ehere will be 30% and the remaining 70% will be used to train our model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

# create an object/instance for Linear Regression model
# y = mx+b
lm = LinearRegression()

# train/fit the data using the Linear Regression object 
lm.fit(X_train, y_train)

# Print the coeffecients that was calculated by the model for each feature. 
print("Coeffeceints: " , lm.coef_)

# predict the next y valuse by passing the test variable of features using the model's predict method
predictions = lm.predict(X_test)

# plot a graph between test variable of y and predicted values
plt.scatter(y_test, predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted y')
plt.show()

# evaluate by calculating the MAS, MSE, RMSE of the model using sklearn's metrics
print("\nMAE: ", metrics.mean_absolute_error(y_test, predictions))
print("MSE: ", metrics.mean_squared_error(y_test, predictions))
print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# do a distribution plot to find out under which distribution our dataset and predictions have occurred by calculating the residual data
# if the residual data is normally distributed then the model that we've used for the dataset is a good choice
sns.distplot((y_test-predictions), bins=50)
plt.show()

# use the coeffecients of the features to change hypothesis or infer the problem given 
coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
print(coeffecients)