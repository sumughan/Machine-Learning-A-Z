# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 06:17:13 2018

@author: SARAVINDAN
"""

#Polinomial Regression

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
#1:2 is used so that X is a matrix. Y should be a vector.
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#simple Linear Regression to compare with Polynomial regression.
from sklearn.linear_model import LinearRegression
linear_reg=LinearRegression()

linear_reg.fit(X,y)

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poli_feature=PolynomialFeatures(degree=4)

X_poli = poli_feature.fit_transform(X)

linear_reg2 = LinearRegression()
linear_reg2.fit(X_poli,y)

#visualize Linear Regression
plt.scatter(X,y,color='red')
plt.plot(X,linear_reg.predict(X),color='blue')
plt.title('Linear Regression plot original vs predict')
plt.xlable('Position Level')
plt.ylable('Salary')

#visualize Polynomial Regression
plt.scatter(X,y,color='red')
plt.plot(X,linear_reg2.predict(X_poli),color='blue')
plt.title('Polynomial Regression plot original vs predict')
plt.xlable('Position Level')
plt.ylable('Salary')

#Predict Linear regression output
linear_reg.predict(6.5)

#Predict Polinomial Regression
linear_reg2.predict(poli_feature.fit_transform(6.5))