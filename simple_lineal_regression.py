import numpy as np
import pandas as pd
from gradient_descent import *
from coordinate_descent import *


def mean(x):
    return sum(x)/len(x)

def snn(df,x,y):
    x_mean=np.mean(df[x].values)
    y_mean=np.mean(df[y].values)
    x_vector=df[x].values
    y_vector=df[y].values
    xsubs = x_vector-x_mean
    sxx=np.sum(xsubs)**2
    sxy=np.sum((y_vector-y_mean)) * np.sum(xsubs)
    return sxx, sxy

def predict_SL(valor,a,b):
    return a*valor+b


from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
rs = RandomState(MT19937(SeedSequence(123456789)))

x=np.random.randint(60,500,500)
y= 4*x+1200

df=pd.DataFrame({'x':x,'y':y})
sxx,sxy = snn(df,'x','y')
a = np.divide(sxy,sxx)
b = np.mean(df['y'].values) - a*np.mean(df['x'].values)

valor=int(input("Ingrese el area de su casa"))
linear_mse=predict_SL(valor,a,b)

weights_gd = batch_gradient_descent(df['x'].values,df['y'].values,learning_rate=0.00001,n_iter=800000,regression_type='linear_regression')
linear_gd = weights_gd[0]+weights_gd[1]*valor

weights_coord = coordinate_descent(df['x'].values,df['y'].values,learning_rate=0.00001,n_iter=800000,regression_type='linear_regression')
linear_cd= weights_coord[0]+weights_coord[1]*valor



print("El precio de su con gd es: ",linear_gd)
print("El precio de su casa con coordinate descent es: ",linear_cd)
print(weights_coord)
print(weights_gd)