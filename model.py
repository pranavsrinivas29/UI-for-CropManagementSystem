
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
import json

# Importing the dataset
data = pd.read_csv(r"Fertilizer Prediction.csv")

cols=data.shape[1]
label= pd.get_dummies(data.FertilizerName).iloc[: , 1:]
data= pd.concat([data,label],axis=1)


data.drop('FertilizerName', axis=1,inplace=True)
data.drop('SoilType', axis=1,inplace=True)
data.drop('CropType', axis=1,inplace=True)

X=data.iloc[:, 0:6].values
y=data.iloc[: ,6:].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

# Fitting Simple Linear Regression to the Training set

from sklearn.ensemble import RandomForestClassifier
algo=RandomForestClassifier(n_estimators=10,random_state=10)

algo.fit(X_train,y_train)
# Predicting the Test set results
y_pred = algo.predict(X_test)

# Saving model using pickle
pickle.dump(algo, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load( open('model.pkl','rb'))
#print(model.predict([[1.8]]))