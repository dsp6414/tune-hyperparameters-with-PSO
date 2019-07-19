import pandas as pd
import numpy as np
from model_PSO import model
from searchtype import choice, uniform, loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB,ComplementNB,BernoulliNB
from sklearn.svm import NuSVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

if __name__=='__main__':
    X_train=pd.read_csv('X_train_nofilled.csv')
    y_train=pd.read_csv('y_train.csv')
    X_test=pd.read_csv('X_test_nofilled.csv')

    
    index=np.random.permutation(len(X_train.index))
    X_train=X_train[index]
    y_train=y_train[index]

    X_train_pre=StandardScaler.fit_transform(X_train)  
    X_test_pre=StandardScaler.transform(X_test)



    
    model_list=['LogisticRegression','GaussianNB','ComplementNB','BernoulliNB','NuSVC','MLPClassifier','KNeighborsClassifier']
#    model_list=model_list[7:8]
    
    param_dict={}

    param_dict['LogisticRegression']={
            'C':[loguniform,-5,5],
            'class_weight':[choice,0,2,['balanced','None']],
            'warm_start':[choice,0,2,[True,True]]
            }

    param_dict['GaussianNB']={
            'var_smoothing':[choice,0,2,[10**-9,10**-9]]
            }

    param_dict['ComplementNB']={
            'alpha':[choice,0,2,[1,1]]
            }

    param_dict['BernoulliNB']={
            'alpha':[choice,0,2,[1,1]]
            }

    param_dict['NuSVC']={
            'nu':[uniform,0,1],
            'kernel':[choice,0,4,['linear','rbf','sigmoid','precomputed']],
            'coef0':[loguniform,-2,2],
            'gamma':[uniform,0,10],
            'probability':[choice,0,2,['True','True']],
            'class_weight':[choice,0,2,['balanced',None]]            
            }
    
    param_dict['MLPClassifier']={
            'hidden_layer_sizes':[choice,0,4,[(10,5),(20,10),(10,5,3),(20,10,5)]],
#            'activation':[choice,0,2,['identity','relu']],
            'alpha':[loguniform,-5,-1],
#            'learning_rate':[choice,0,2,['constant','adaptive']],
            'learning_rate_init':[loguniform,-5,-1],
#            'max_iter':[choice,0,2,[1000,2000]],
#            'beta_1':[uniform,0.8,1],
#            'beta_2':[uniform,0.9,1],           
            'warm_start':[choice,0,2,[True,True]]
            }
 
    param_dict['KNeighborsClassifier']={
        'weights':[choice,0,2,['uniform','distance']]
            }

    
    model_dict={}
    model_dict['LogisticRegression']=LogisticRegression
    model_dict['GaussianNB']=GaussianNB
    model_dict['ComplementNB']=ComplementNB
    model_dict['BernoulliNB']=BernoulliNB  
    model_dict['NuSVC']=NuSVC
    model_dict['MLPClassifier']=MLPClassifier   
    model_dict['KNeighborsClassifier']=KNeighborsClassifier
    model_dict['XGBClassifier']=XGBClassifier
    
    for model_name in model_list:       
        clf_model=model()
        clf_model.train(X_train_pre,y_train,model_dict[model_name],param_dict[model_name])
        clf_model.predict(X_test_pre)