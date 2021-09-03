#%%
# prediction of house based on area, bedroom and age
import pandas as pd
import numpy as np
from sklearn import linear_model
import math
df = pd.read_csv("data3.csv")
median_bedrooms = math.floor(df.bedrooms.median())
median_bedrooms
df.bedrooms = df.bedrooms.fillna(median_bedrooms)
df
reg = linear_model.LinearRegression()
reg.fit(df[['area','bedrooms', 'age']], df.price)
reg.predict([[3000,7,15]])

# saving our model into a file
import pickle
with open('model_pickle','wb') as f:
    pickle.dump(reg,f)
with open('model_pickle','rb') as f:
    mp = pickle.load(f)
mp.predict([[5000, 4, 30]])

# %%
