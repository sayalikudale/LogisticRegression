
# coding: utf-8

# In[1]:

import scipy.io
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from statistics import mean 


# In[2]:

# load data 
heart_Data = scipy.io.loadmat('Data/heart.mat')
heart_attributes=heart_Data['dat']
heart_label=heart_Data['label']


import warnings
warnings.filterwarnings('ignore')


# In[3]:

def featureSubsetScore(heartMatSub,Label):
    test_scores=list()
    train_scores=list()

    for seed in range(1, 11):
        X_train, X_test, y_train, y_test = train_test_split(heartMatSub, Label, 
                                                        test_size=0.2592, random_state=seed)
        
        # fit a model
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # predict labels on train set
        predProb = model.predict_proba(X_train)

        threshold = 0.5
        trainPred = ( predProb[:, 1 ] > threshold ) + 1;  

        ntrainCorrect=0
        for i in range(0, len(trainPred)):
            if (trainPred[i] == y_train.iloc[i].Result):
                ntrainCorrect+=1
        
        train_scores.append((ntrainCorrect)/len(y_train))
        

        # predict labels on test set
        predProbTest = model.predict_proba(X_test)
        testPred = ( predProbTest[:, 1 ] > threshold ) + 1;  
        
        ntestCorrect=0
        for i in range(0, len(testPred)):
            if (testPred[i] == y_test.iloc[i].Result):
                ntestCorrect+=1
              
        test_scores.append((ntestCorrect)/len(y_test))

        
        
    trainScoreMean=mean(train_scores)
    testScoreMean=mean(test_scores)
    
   
    return trainScoreMean,testScoreMean
    


# In[7]:

# Execute the script

trainScoreMean,testScoreMean=featureSubsetScore(pd.DataFrame(heart_attributes[:, 0:12]),pd.DataFrame(heart_Data['label'],columns = ['Result']))

print('Training DataSet Mean : %s' % trainScoreMean)
print('Testing DataSet Mean  : %s' % testScoreMean)


# In[ ]:



