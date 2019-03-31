# Import initial set of libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read input data
input_data = pd.read_csv("Churn_Modelling.csv")


#Split between features and measures

X=input_data.iloc[:,3:13]
y=input_data.iloc[:,13]

#Lable encoding to convert text to numbers and OneHot Encoding to make them columns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

le_country = LabelEncoder()

X['Geography']=le_country.fit_transform(X['Geography'])

le_gender = LabelEncoder()
X['Gender']=le_gender.fit_transform(X['Gender'])

ohe_country = OneHotEncoder(categorical_features=[1])

X=ohe_country.fit_transform(X).toarray()

#Removing one of the OneHot columns so as to avoid issue with all columns being present
X=X[:,1:]

#Train test split before machine learning.
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.2,random_state=0)


#Scaling the features before neural Networks are applied
from sklearn.preprocessing import StandardScaler

st_scaler = StandardScaler()

X_train = st_scaler.fit_transform(X=X_train)
X_test = st_scaler.transform(X=X_test)

#Import functions for ANN modeling
import keras
from keras.models import Sequential
from keras.layers import Dense

#Create Model and add layers to it. We will have one input, one hidden and one output layer
model = Sequential()

#Adding input layer. There are 11 input indepnedent variables
model.add(Dense(units=6, kernel_initializer='uniform', activation='relu',input_dim=11))

#Adding hidden layer
model.add(Dense(units=6,activation='relu',kernel_initializer='uniform'))

#Adding output layer with Sigmoid activation function to get % probability.
model.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))

#Compile the ANN model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Fit model
model.fit(x=X_train,y=y_train,batch_size=10,epochs=100)

#Predict model
y_pred=model.predict(x=X_test)
y_pred=(y_pred>.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)