

import pylab
import calendar
import numpy as np
import pandas as pd
import seaborn as sn
from scipy import stats
#import missingno as msno
from datetime import datetime
import matplotlib.pyplot as plt
import pickle


df= pd.read_csv("raw_wheat.csv")


df=df.rename(index=str, columns={"Sl no.": "Sl_no", "Min Price (Rs./Quintal)": "Min_Price","Price Date":"Price_Date","Modal Price (Rs./Quintal)":"Modal_Price"})

df.Price_Date = pd.to_datetime(df.Price_Date, errors='coerce')

df=df.sort_values(by='Price_Date')
df.drop_duplicates('Price_Date', inplace = True)

df.to_csv("ww.csv", sep=',')

import itertools
import statsmodels.api as sm

fields = ['Modal_Price', 'Price_Date']
df= pd.read_csv("ww.csv",skipinitialspace=True, usecols=fields)
df.Price_Date = pd.to_datetime(df.Price_Date, errors='coerce')

df=df.set_index('Price_Date')

data = df.copy()
y = data

y = y['Modal_Price'].resample('MS').mean()

y = y.fillna(y.bfill())


p = d = q = range(0, 2)  # Define the p, d and q parameters to take any value between 0 and 2

# Generate all different combinations of p, q and q triplets
#p is the number of autoregressive terms,
#d is the number of nonseasonal differences needed for stationarity, and
#q is the number of lagged forecast errors in the prediction equation.
pdq = list(itertools.product(p, d, q))

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in pdq]

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue
        


pred_dynamic = results.get_prediction(start=pd.to_datetime('2018-01-01'), dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()

# Get forecast 70 steps ahead in future
pred_uc = results.get_forecast(steps=70)

# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()
    
modified = pred_ci.reset_index()
modified.rename(columns = {'index':'Date'}, inplace = True)
    
#l=modified.loc[modified['Date'] == inp]
pickle.dump(results, open('model3.pkl','wb'))

# Loading model to compare the results
model3 = pickle.load( open('model3.pkl','rb'))