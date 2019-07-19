import numpy as np

import pyswarms as ps
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,make_scorer
from sklearn.externals import joblib
import pandas as pd
import time
import datetime
   
def ErrorDistribs(y_true,y_pred):
    return abs(y_true-y_pred)/y_true    
    
def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3

auc_scorer = make_scorer(tpr_weight_funtion)

class model():
    def __init__(self,n_particles=10,c1=0.5,c2=0.5,w=0.9,verbose=2,cv=5,scoring=auc_scorer):
        self.n_particles=n_particles
        self.c1=c1
        self.c2=c2
        self.w=w
        self.options={'c1':self.c1,'c2':self.c2,'w':self.w}
        self.verbose=verbose
        self.cv=cv
        self.scoring=auc_scorer
        
    def train(self,X_train,y_train,clf,param_distribs):
        
        start=time.perf_counter()
        
        self.X_train=X_train
        self.y_train=y_train
        self.clf=clf
        self.param_distribs=param_distribs
        self.dimensions=len(param_distribs)
        
        upper=np.zeros(self.dimensions)
        lower=np.zeros(self.dimensions)
        
        for count, (key, value) in enumerate(self.param_distribs.items()):
            lower[count]=value[1]
            upper[count]=value[2]
        
        bounds=(lower,upper)

        optimizer=ps.single.GlobalBestPSO(n_particles=self.n_particles,dimensions=self.dimensions,options=self.options,bounds=bounds)
        best_cost,best_pos=optimizer.optimize(self.search,iters=25,verbose=self.verbose,print_step=10)       

#        best_pos=[-0.7811003950341757, 4.736212131795903, 0.3134303131418766]
        self.best_params={}
        
        for count, (key, value) in enumerate(self.param_distribs.items()):
            if value[0].__name__=='choice':
                index=value[0](best_pos[count])
                self.best_params[key]=value[3][index]                    
            else:
                self.best_params[key]=value[0](best_pos[count])

        self.final_model=self.clf(**self.best_params)
        self.final_model.fit(self.X_train,self.y_train)

        
        now=time.perf_counter()

        
#        with open('{}.txt'.format(self.clf.__name__),'w+') as f:
#            f.write('RMSE: {}\r\nR2: {}\r\nError_median: {}\r\n'.format(RMSE,R2,error_median))
        
        self.my_dict={}
        self.my_dict['train_time']=now-start
        self.my_dict['auc_score']=-best_cost
#        self.my_dict['y_true']=self.y_train.ravel()
#        self.my_dict['y_pred']=y_pred.ravel()
        df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in self.my_dict.items() ]))
        df.to_excel('{}.xlsx'.format(self.clf.__name__))        

        joblib.dump(self.final_model,'{}.pkl'.format(self.clf.__name__))
#        my_model_loaded=joblib.load('{}.pkl'.format(self.clf.__name__))

    def search(self,param):
        score_array=np.zeros((self.n_particles,self.cv))
        fit_params={}
        
        for i in range(self.n_particles):

            for count, (key, value) in enumerate(self.param_distribs.items()):
                if value[0].__name__=='choice':
                    index=value[0](param[i,count])
                    fit_params[key]=value[3][index]                    
                else:
                    fit_params[key]=value[0](param[i,count])
#            cv=ShuffleSplit(n_splits=5,test_size=0.3)
            score_array[i,:]=cross_val_score(self.clf(**fit_params),self.X_train,self.y_train,scoring=self.scoring,cv=self.cv)
        return -np.mean(score_array,axis=1)
            
    def predict(self,X_test):
        """
        x: numpy.ndarray of shape (n_particles, dimensions)  
        """
        

        y_pred=self.final_model.predict_proba(X_test)
        now = datetime.datetime.now()
        now = now.strftime('%m-%d-%H-%M')
        sub=pd.read_csv('sub.csv')
        sub['Tag']=y_pred
        sub.to_csv('%s_%s.csv' %(self.clf.__name__, now),index=False)
        
        
        
        
        return y_pred
    





        
        