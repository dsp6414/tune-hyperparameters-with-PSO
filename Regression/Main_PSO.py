import pandas as pd
import numpy as np
from model_PSO import model
from sklearn.svm import SVR
from searchtype import choice, uniform, loguniform
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from quadratic_regression import QuadraticRegression
from sklearn.neural_network import MLPRegressor


if __name__=='__main__':
    data=pd.read_csv('POP_1.txt',sep=' ',header=None,na_filter=False,usecols=np.arange(72))
    X_train=data.values[:200,:]
    data=pd.read_csv('MEM_1.txt',sep=' ',header=None,na_filter=False,usecols=np.arange(1))
    y_train=data.values.ravel()[:200]
    
    index=np.random.permutation(200)
    X_train=X_train[index]
    y_train=y_train[index]
    
    model_list=['Ridge','QuadraticRegression','DecisionTreeRegressor','SVR','MLPRegressor','RandomForestRegressor','GradientBoostingRegressor']
#    model_list=model_list[0:1]
    
    param_dict={}

    param_dict['Ridge']={
            'alpha':[uniform,0,1]
            }

    param_dict['QuadraticRegression']={
            'alpha':[uniform,0,1]
            }

    param_dict['DecisionTreeRegressor']={
            'criterion':[choice,0,2,['mse','friedman_mse']]
            }

    param_dict['SVR']={
            'kernel':[choice,0,2,['linear','rbf']],
            'C':[loguniform,-5,5],
            'gamma':[uniform,0,10]
            }
    
    param_dict['MLPRegressor']={
            'hidden_layer_sizes':[choice,0,4,[(10,10),(20,20),(10,10,10),(20,20,20)]],
            'activation':[choice,0,2,['identity','relu']],
            'alpha':[loguniform,-5,-1],
            'learning_rate':[choice,0,2,['constant','adaptive']],
            'learning_rate_init':[loguniform,-5,-1],
#            'max_iter':[choice,0,2,[1000,2000]],
#            'beta_1':[uniform,0.8,1],
#            'beta_2':[uniform,0.9,1]            
            }
 
    param_dict['RandomForestRegressor']={
        'n_estimators':[choice,0,2,[10,100]]
            }

    param_dict['GradientBoostingRegressor']={
        'loss':[choice,0,3,['ls','lad','huber']],
        'learning_rate':[loguniform,-5,-1],
        'max_depth':[choice,0,3,[5,10,15]],
        'alpha':[uniform,0,1]
            }    
    
    model_dict={}
    model_dict['Ridge']=Ridge
    model_dict['QuadraticRegression']=QuadraticRegression
    model_dict['DecisionTreeRegressor']=DecisionTreeRegressor
    model_dict['SVR']=SVR   
    model_dict['MLPRegressor']=MLPRegressor
    model_dict['RandomForestRegressor']=RandomForestRegressor   
    model_dict['GradientBoostingRegressor']=GradientBoostingRegressor
    
    for model_name in model_list:       
        reg_model=model()
        reg_model.train(X_train,y_train,model_dict[model_name],param_dict[model_name])
        reg_model.optimize()