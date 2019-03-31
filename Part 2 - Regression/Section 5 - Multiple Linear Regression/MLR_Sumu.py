import pandas as pd

input_data=pd.read_csv("50_Startups.csv")

print(input_data.head())

import matplotlib.pyplot as plt
import seaborn as sns

#sns.pairplot(input_data)
#plt.show()

X=input_data.iloc[:,:-1].values
y=input_data.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder