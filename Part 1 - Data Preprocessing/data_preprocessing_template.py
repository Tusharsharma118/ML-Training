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
"""
# Filling in Missing Data
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler

# Creating object of imputer for missing Data
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,1:3]) # which rows of which matrix to fit it on
X[:,1:3] = imputer.transform(X[:,1:3])  # apply the transformation

# create Label encoder to parse categorical data
labelEncoder_X = LabelEncoder()
labelEncoder_Y = LabelEncoder()
# Encode Country
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
#Encode Prediction(Purchase)
Y = labelEncoder_Y.fit_transform(Y)
# Need to dummy encode X to prevent comparisons within the array
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X= oneHotEncoder.fit_transform(X).toarray()
"""
# Splitting data into Testset and Training Set
from sklearn.model_selection import train_test_split

# Splitting dataset
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
												# 		1:3 ratio for X:Y; random_state so that value of random split # #   	remains same for all
"""
#Feature  Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train) #Scaling Training Data
X_test = sc_X.transform(X_test)       #Scaling Test Data
"""

