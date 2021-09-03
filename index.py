#%%
# prediction of price of land based on area...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df = pd.read_csv("data2.csv")
df
# print(df.to_string())
# %matplotlib inline
plt.xlabel('Area')
plt.ylabel('Price')
plt.plot(df.area, df.price, color='red',marker='+')
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)
reg.predict([[5000]])
# %%
