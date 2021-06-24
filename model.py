
import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import ExtraTreesRegressor
#from sklearn.linear_model import LinearRegression

dt=pd.read_csv("Forest_fire.csv")

x = dt[['Oxygen','Temperature','Humidity']]
y = dt['Fire Occurence']
x_train,x_test,y_train,y_test= train_test_split(x,y ,test_size = 0.25,random_state= 42)
regressor = ExtraTreesRegressor(n_estimators = 10, random_state = 0)
regressor.fit(x, y) 

# Saving model to disk
pickle.dump(regressor, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[8.2, 51]]))
