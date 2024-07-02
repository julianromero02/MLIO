import numpy as np
import pandas as pd
from gradient_descent import *

x=np.random.randint(1,100,100)
y= x**2+100

df=pd.DataFrame({'x':x,'y':y})
def mean(x):
    return sum(x)/len(x)

def snn(df,x,y):
    x_mean=mean(df[x])
    y_mean=mean(df[y])
    x_vector=df['x'].values
    y_vector=df['y'].values
    xsubs = x_vector-x_mean
    sxx=np.sum(xsubs)**2
    sxy=np.sum((y_vector-y_mean)) * np.sum(xsubs)
    return sxx, sxy

sxx,sxy = snn(df,'x','y')
a = sxy/sxx
b = mean(df['y']) - a*mean(df['x'])

def predict_SL(valor,a,b):
    return a*valor+b
valor=int(input("Ingrese el area de su casa"))
linear_mse=predict_SL(valor,a,b)
gradient = stochastic_gradient_descent(df[['x']].values,df['y'].values,learning_rate=0.01,n_iter=1000,regression_type='linear_regression')
linear_gd= gradient*valor


print("El precio de su casa es: ",linear_mse)


print("El precio de su casa con Gradient Descent es: ",linear_gd)
print("A")