import numpy as np
import pandas as pd

input=pd.read_csv("Salary_Data.csv")

X=input.iloc[:,:-1].values
y=input.iloc[:,1].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y)

from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

import matplotlib.pyplot as plt
import seaborn as sns

#sns.pairplot(X_train,y_train)

plt.scatter(X_test,y_test,color='Red')
plt.plot(X_test,model.predict(X_test))
#plt.show()

#sns.pairplot(input)

#iris = sns.load_dataset("iris")
#sns.pairplot(iris);

#sns.distplot(X,rug=True)

#sns.jointplot(x="YearsExperience",y="Salary",data=input)
#sns.regplot(x=X,y=model.predict(y),scatter=False)

#sns.regplot(x=X, y=y, fit_reg=False, scatter_kws={"color": "green"});
#sns.regplot(x=X, y=model.predict(y),scatter=False, scatter_kws={"color": "green"});
plt.show()
print(input.head())