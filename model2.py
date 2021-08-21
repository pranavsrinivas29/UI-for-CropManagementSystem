import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import pickle

warnings.filterwarnings('ignore')
df = pd.read_csv(r"cpdata.csv")

cols=df.shape[1]
X=df.iloc[:, 0:4].values
y=df.iloc[: ,4:].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

from sklearn.ensemble import RandomForestClassifier
algo2=RandomForestClassifier(n_estimators=10,random_state=10)

algo2.fit(X_train,y_train)


y_pred = algo2.predict(X_test)

pickle.dump(algo2, open('model2.pkl','wb'))


model2 = pickle.load( open('model2.pkl','rb'))