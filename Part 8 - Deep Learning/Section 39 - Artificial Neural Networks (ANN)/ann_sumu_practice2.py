#Practice for ANN to build the myelin layer

import pandas as pd
import numpy as np

#read file
input_data = pd.read_csv("/Users/SAravindan/OneDrive - Brunswick Corporation/Pers/Tech/ML/SuperDataScience/Machine Learning A-Z/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Churn_modelling.csv")

#inspect the variable and split it into train and test set
X=input_data.iloc[:,3:13]
y=input_data.iloc[:,13]

#Encoding Categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

geo_le = LabelEncoder()
gender_le = LabelEncoder()

X['Geography']=geo_le.fit_transform(X['Geography'])
X['Gender']=gender_le.fit_transform(X['Gender'])

geo_ohe = OneHotEncoder(categorical_features=[1])
X=geo_ohe.fit_transform(X).toarray()

#Remove first value
X=X[:,1:]

#Train test split
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)

from sklearn.preprocessing import StandardScaler
st_scaler = StandardScaler()

X_train=st_scaler.fit_transform(X_train)
X_test=st_scaler.transform(X_test)

from keras.models import Sequential
from keras.layers import Dense

model=Sequential()

#Adding layers
model.add(Dense(6,activation='relu',kernel_initializer='uniform',input_dim=11))

#add hidden layer
model.add(Dense(6,activation='relu',kernel_initializer='uniform'))

#add output layer
model.add(Dense(1,activation='sigmoid',kernel_initializer='uniform'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fit model
model.fit(X_train,y_train,batch_size=10,epochs=100)

y_predict = model.predict(X_test)

y_pred = (y_predict>=.5)

y_test.shape

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)