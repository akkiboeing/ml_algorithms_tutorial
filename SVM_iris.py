import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

iris = sns.load_dataset('iris')

print(iris.head())

sns.pairplot(iris, hue = 'species', palette = 'Dark2')
plt.show()

setosa = iris[iris['species']=='setosa']
sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'], cmap="plasma", shade=True, shade_lowest=False)
plt.show()

X = iris.drop('species', axis = 1)
y = iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

model = SVC()
model.fit(X_train, y_train)
pred = model.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))

param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose = 2)
grid.fit(X_train, y_train)
grid_pred = grid.predict(X_test)

print(confusion_matrix(y_test, grid_pred))
print(classification_report(y_test, grid_pred))