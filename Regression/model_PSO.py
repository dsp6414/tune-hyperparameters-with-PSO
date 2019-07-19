import numpy as np
from sklearn.preprocessing import StandardScaler
import pyswarms as ps
from sklearn.model_selection import cross_val_score,ShuffleSplit
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.externals import joblib
import pandas as pd
import time

def discretize(x,num):
    result = min(num-1, max(0, x))
    return int(result)    
    
def ErrorDistribs(y_true,y_pred):
    return abs(y_true-y_pred)/y_true    
    

class model():
    def __init__(self,n_particles=20,c1=0.5,c2=0.5,w=0.9,verbose=1,cv=5,scoring='neg_mean_squared_error'):
        self.StandardScaler=StandardScaler()
        self.n_particles=n_particles
        self.c1=c1
        self.c2=c2
        self.w=w
        self.options={'c1':self.c1,'c2':self.c2,'w':self.w}
        self.verbose=verbose
        self.cv=cv
        self.scoring=scoring
        
    def train(self,X_train,y_train,reg,param_distribs):
        
        start=time.perf_counter()
        
        self.X_train_pre=self.StandardScaler.fit_transform(X_train)        
        self.y_train=y_train
        self.reg=reg
        self.param_distribs=param_distribs
        self.dimensions=len(param_distribs)
        
        upper=np.zeros(self.dimensions)
        lower=np.zeros(self.dimensions)
        
        for count, (key, value) in enumerate(self.param_distribs.items()):
            lower[count]=value[1]
            upper[count]=value[2]
        
        bounds=(lower,upper)

        optimizer=ps.single.GlobalBestPSO(n_particles=self.n_particles,dimensions=self.dimensions,options=self.options,bounds=bounds)
        best_cost,best_pos=optimizer.optimize(self.search,iters=50,verbose=self.verbose)       

#        best_pos=[-0.7811003950341757, 4.736212131795903, 0.3134303131418766]
        self.best_params={}
        
        for count, (key, value) in enumerate(self.param_distribs.items()):
            if value[0].__name__=='choice':
                index=value[0](best_pos[count])
                self.best_params[key]=value[3][index]                    
            else:
                self.best_params[key]=value[0](best_pos[count])

        self.final_model=self.reg(**self.best_params)
        self.final_model.fit(self.X_train_pre,self.y_train)
        y_pred=self.final_model.predict(self.X_train_pre)
        
        now=time.perf_counter()
        
        RMSE=np.sqrt(mean_squared_error(y_train,y_pred))/10**6
        R2=r2_score(y_train,y_pred)
        rela_error=ErrorDistribs(y_train,y_pred)*100
        error_hist,_=np.histogram(rela_error,bins=50)
        error_median=np.median(rela_error)
        
#        with open('{}.txt'.format(self.reg.__name__),'w+') as f:
#            f.write('RMSE: {}\r\nR2: {}\r\nError_median: {}\r\n'.format(RMSE,R2,error_median))
        
        self.my_dict={}
        self.my_dict['train_time']=now-start
        self.my_dict['RMSE']=RMSE
        self.my_dict['R2']=R2
        self.my_dict['Error_Median']=error_median
        self.my_dict['Error_Hist']=error_hist.ravel()
        self.my_dict['y_true']=self.y_train.ravel()
        self.my_dict['y_pred']=y_pred.ravel()
        

        joblib.dump(self.final_model,'{}.pkl'.format(self.reg.__name__))
#        my_model_loaded=joblib.load('{}.pkl'.format(self.reg.__name__))

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
            score_array[i,:]=cross_val_score(self.reg(**fit_params),self.X_train_pre,self.y_train,scoring=self.scoring,cv=self.cv)
        return -np.mean(score_array,axis=1)
            
    def predict(self,X_test):
        """
        x: numpy.ndarray of shape (n_particles, dimensions)  
        """
        
        X_test_pre=self.StandardScaler.transform(X_test)
        y_pred=self.final_model.predict(X_test_pre)
        return -y_pred
    
    def optimize(self):
        
        start=time.perf_counter()
        
        upper=250*np.ones(72)
        lower=15*np.ones(72)
        bounds=(lower,upper)
        
        optimizer=ps.single.GlobalBestPSO(n_particles=self.n_particles,dimensions=72,options=self.options,bounds=bounds)
        best_cost,best_pos=optimizer.optimize(self.predict,iters=100,verbose=self.verbose)
        best_pos=np.array(best_pos)
        
        cost_history=np.array(optimizer.cost_history)
        
        now=time.perf_counter()
        
        self.my_dict['opt_time']=now-start
        self.my_dict['X_test']=best_pos.ravel()
        self.my_dict['cost_history']=cost_history.ravel()
        
        df=pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in self.my_dict.items() ]))
        df.to_excel('{}.xlsx'.format(self.reg.__name__))

#        while True:
#            try:
#                my_model_loaded=joblib.load('{}.pkl'.format(self.reg.__name__))
#                print(my_model_loaded)
#            except EOFError:
#                break


        
        