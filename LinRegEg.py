import numpy as np
from sklearn import linear_model, datasets, tree
import matplotlib.pyplot as plt

number_of_samples = 100
x = np.linspace(-np.pi, np.pi, number_of_samples)
y = 0.5*x+np.sin(x)+np.random.random(x.shape)
plt.scatter(x,y,color='black') #Plot y-vs-x in dots
plt.xlabel('x-input feature')
plt.ylabel('y-target values')
plt.title('Data for linear regression')
plt.show()

#Linear Regression
random_indices = np.random.permutation(number_of_samples)

#Training Set
x_train = x[random_indices[:70]]
y_train = y[random_indices[:70]]

#Validation Set
x_validate = x[random_indices[70:85]]
y_validate = y[random_indices[70:85]]

#Test Set
x_test = x[random_indices[:85]]
y_test = y[random_indices[:85]]

model = linear_model.LinearRegression() #Create a Least squared error Linear Regression object

#ScikitLearn takes inputs as matrices. Arrays are reshaped into matrices.
x_train_for_line_fit = np.matrix(x_train.reshape(len(x_train),1))
y_train_for_line_fit = np.matrix(y_train.reshape(len(y_train),1))

#Fit the line to the training data
model.fit(x_train_for_line_fit,y_train_for_line_fit)

#Plotting the line
plt.scatter(x_train, y_train, color='black')
plt.plot(x.reshape((len(x),1)),model.predict(x.reshape((len(x),1))),color='blue')
plt.xlabel('x-input feature')
plt.ylabel('y-target values')
plt.title('Line fit to training data')
plt.show()

#Evaluate the model
mean_val_error = np.mean( (y_validate - model.predict(x_validate.reshape(len(x_validate),1)))**2 )
mean_test_error = np.mean( (y_test - model.predict(x_test.reshape(len(x_test),1)))**2 )

print ('Validation MSE: ', mean_val_error)
print ('Test MSE: ', mean_test_error)

