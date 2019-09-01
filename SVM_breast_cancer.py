import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

cancer = load_breast_cancer()

print(cancer.keys())
print(cancer['DESCR'])
print(cancer['target'])
print(cancer['target_names'])

bc_df = pd.DataFrame(cancer['data'], columns = cancer['feature_names'])

print(bc_df.head())
print(bc_df.info())
print(bc_df.describe())

X = bc_df
y = cancer['target']
X_train, X_test, y_train, y_test = train_test_split(X , y, test_size = 0.3, random_state = 101)

model = SVC()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(confusion_matrix(y_test, pred))
print('\n')
print(classification_report(y_test, pred))

param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose = 3)

grid.fit(X_train, y_train)

grid.best_params_
print(grid.best_params_)
grid.best_estimator_
print(grid.best_estimator_)

grid_pred = grid.predict(X_test)

print('\n')
print(confusion_matrix(y_test, grid_pred))
print('\n')
print(classification_report(y_test, grid_pred))