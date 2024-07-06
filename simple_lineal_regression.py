import numpy as np
import pandas as pd
from gradient_descent import *
from coordinate_descent import *
from regressions import ridge_regression, lasso_regression, elastic_net

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

x=np.random.randint(60,500,(10000,2))
y= 4*x[:,0]+10*x[:,1]+1200

#df=pd.DataFrame({'x':x[0],'y':y})
#sxx,sxy = snn(df,'x','y')
#a = np.divide(sxy,sxx)
#b = np.mean(df['y'].values) - a*np.mean(df['x'].values)

valor1=int(input("Ingrese el area de su casa"))
valor2 = int(input("Ingrese el segundo area de su casa"))
#linear_mse=predict_SL(valor,a,b)
lr = 1e-09
epoch = 1000000
weights_gd = batch_gradient_descent(x,y,learning_rate=lr,n_iter=epoch,regression_type='linear_regression')
linear_gd = weights_gd[0]+weights_gd[1]*valor1+weights_gd[1]*valor2

weights_coord = coordinate_descent(x,y,learning_rate=0.00001,n_iter=epoch,regression_type='linear_regression')
linear_cd= weights_coord[0]+weights_coord[1]*valor1+weights_coord[2]*valor2

weights_ridge_gd = ridge_regression(x,y,learning_rate=lr,n_iter=epoch, gradient_type='gradient_descent')

weights_ridge_cd = ridge_regression(x,y,learning_rate=0.00001,n_iter=epoch, gradient_type='coordinate_descent')

weights_lasso_gd = lasso_regression(x,y,learning_rate=lr,n_iter=epoch, gradient_type='gradient_descent')
weights_lasso_cd = lasso_regression(x,y,learning_rate=0.00001,n_iter=epoch, gradient_type='coordinate_descent')

weights_elastic_gd = elastic_net(x,y,learning_rate=lr,n_iter=epoch, gradient_type='gradient_descent')
weights_elastic_cd = elastic_net(x,y,learning_rate=0.00001,n_iter=epoch, gradient_type='coordinate_descent')

print("El precio real es: ",linear_gd)

print("El precio de su con gd es: ",linear_gd)
print("El precio de su casa con coordinate descent es: ",linear_cd)
print("El precio de su casa con ridge coordinate descent es: ", weights_ridge_cd[0]+weights_ridge_cd[1]*valor1+weights_ridge_cd[2]*valor2)
print("El precio de su casa con ridge gd es: ",linear_cd)
print("El precio de su casa con lasso gd es: ",linear_cd)
print(weights_coord)
print(weights_gd)