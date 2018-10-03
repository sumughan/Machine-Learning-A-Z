# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 15:48:39 2018

@author: SARAVINDAN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as pt

dataset = pd.read_csv('Data.csv')

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

#Impute data
from sklearn.preprocessing import Imputer

imp=Imputer(missing_values= "NaN",strategy = "mean", axis = 0)

imp=imp.fit(X[:,1:3])

X[:,1:3]=imp.transform(X[:,1:3])

# Encoding data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

labelencoder_X=LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])

onehotencoder = OneHotEncoder(categorical_features = [0])
X=onehotencoder.fit_transform(X).toarray()

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

#Training set test set split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state = 0)

#Scaling values
from sklearn.preprocessing import StandardScaler

sc_X =StandardScaler()

X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

