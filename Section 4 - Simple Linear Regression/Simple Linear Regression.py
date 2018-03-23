# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 10:54:23 2018

@author: tsharma
"""

# Data Preprocessing


##Importing all dependencies
###for mathematics
import numpy as np
### for plots
import matplotlib.pyplot as plt
### for ML
import pandas as pd
## Reading in Data
dataset = pd.read_csv('Salary_Data.csv')
## Loading Feature Vector/Matrix
X = dataset.iloc[:,:-1].values
## Loading Prediction vector
Y = dataset.iloc[:,-1].values

# Splitting data into Testset and Training Set
from sklearn.model_selection import train_test_split

# Splitting dataset
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)


#Importing Linear Regression
from sklearn.linear_model import LinearRegression

# Create Model/Regressor 
regressor = LinearRegression()
# Train regressor on training data
regressor.fit(X_train,Y_train)

# Predicting test sets
y_pred = regressor.predict(X_test)

# Plot Test Data for Visualization
## plotting training data
plt.scatter(X_train,Y_train,color='red')
## Plotting regression Line
plt.plot(X_train,regressor.predict(X_train),color='blue')
## Labels and Names
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Years of Experience vs Salary Plot(TRAINING DATA)')
plt.show()

# Plot Predicted Data over Training set
## plotting test data
plt.scatter(X_test,Y_test,color='red')
## Plotting regression Line
plt.plot(X_test,regressor.predict(X_train),color='blue')
## Labels and Names
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Years of Experience vs Salary Plot(TEST DATA)')
plt.show()



