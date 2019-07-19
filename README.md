# tune-hyperparameters-with-PSO
The hyperparameters of machine learning algorithms are usually tuned by grid search and random search. In the grid search, every combination of the hyperparameters 
is compared to determine the optimum hyperparameter. However, the grid search only searches a small portition of the
hyperparameter space, which are specified subjectively. In the random search, the hyperparameters determined may not even be local optimum. The 
derivative-free optimization method, particle swarm optimization (PSO), is used in this repository to effectively and efficiently find the optimum or near-optimum hyperparameter.
The hyperparameters are categorized as two types: discrete value and continuous value. The continuous value can be selected from uniform distribution if the hyperparameter is in a small range 
or log-uniform distribution if the order of maginitude is unknown. 

For the classifiction, the machine learning algorithms include: logistic regression, SVC, MLP, kNN, naive Bayes.
For the regression, the machine learning algorithms include: linear regression, quadratic regression, SVR, MLP, decision tree, naive Bayes, random forest, gradient boosting machine.

