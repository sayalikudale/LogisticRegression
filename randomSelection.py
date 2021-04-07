
# coding: utf-8

# ## Feature Subset Seletion Script

# In[1]:

#Author: Sayali Kudale

import scipy.io
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from statistics import mean 


# In[2]:

heart_Data = scipy.io.loadmat('Data/heart.mat')
heart_attributes=heart_Data['dat']
heart_label=heart_Data['label']


# In[19]:

import warnings
warnings.filterwarnings('ignore')


# In[5]:

df_heartAttr = pd.DataFrame(heart_attributes, columns = ['Age','Sex','Chest Pain Type','Blood Pressure','Cholestoral','Fblood sugar','ECG','Max heart rate','anigma','oldpeak','slope peak exe','# vessels','thal'])
df_heartLabel = pd.DataFrame(heart_label, columns = ['Result'])


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
    


# In[10]:

#initialization

import random

noOfFeatures=13
randomTrials=1000
FSel=list()
ScoreBest=0
featureSet = [*range(0, 13, 1)] 


# In[24]:


targetNoFeatures=1

for i in range(0,randomTrials):
    
    FTrial =random.sample(featureSet, k=targetNoFeatures)
    trainScore,testScore=featureSubsetScore(pd.DataFrame(heart_attributes[:, FTrial]),df_heartLabel)
    
    if testScore > ScoreBest:
        FSel=FTrial
        ScoreBest=testScore
    
        
print('Selected Subset : %s '%FSel)

print('Best Score of the Subset : %s '%ScoreBest)


# In[25]:

targetNoFeatures=2

for i in range(0,randomTrials):
    
    FTrial =random.sample(featureSet, k=targetNoFeatures)
    trainScore,testScore=featureSubsetScore(pd.DataFrame(heart_attributes[:, FTrial]),df_heartLabel)
    
    if testScore > ScoreBest:
        FSel=FTrial
        ScoreBest=testScore
    
        
print('Selected Subset : %s '%FSel)

print('Best Score of the Subset : %s '%ScoreBest)


# In[26]:

targetNoFeatures=3

for i in range(0,randomTrials):
    
    FTrial =random.sample(featureSet, k=targetNoFeatures)
    trainScore,testScore=featureSubsetScore(pd.DataFrame(heart_attributes[:, FTrial]),df_heartLabel)
    
    if testScore > ScoreBest:
        FSel=FTrial
        ScoreBest=testScore
    
        
print('Selected Subset : %s '%FSel)

print('Best Score of the Subset : %s '%ScoreBest)


# In[27]:

targetNoFeatures=4


for i in range(0,randomTrials):
    
    FTrial =random.sample(featureSet, k=targetNoFeatures)
    trainScore,testScore=featureSubsetScore(pd.DataFrame(heart_attributes[:, FTrial]),df_heartLabel)
    
    if testScore > ScoreBest:
        FSel=FTrial
        ScoreBest=testScore
    
        
print('Selected Subset : %s '%FSel)

print('Best Score of the Subset : %s '%ScoreBest)


# In[28]:

targetNoFeatures=5


for i in range(0,randomTrials):
    
    FTrial =random.sample(featureSet, k=targetNoFeatures)
    trainScore,testScore=featureSubsetScore(pd.DataFrame(heart_attributes[:, FTrial]),df_heartLabel)
    
    if testScore > ScoreBest:
        FSel=FTrial
        ScoreBest=testScore
    
        
print('Selected Subset : %s '%FSel)

print('Best Score of the Subset : %s '%ScoreBest)


# In[29]:

targetNoFeatures=6


for i in range(0,randomTrials):
    
    FTrial =random.sample(featureSet, k=targetNoFeatures)
    trainScore,testScore=featureSubsetScore(pd.DataFrame(heart_attributes[:, FTrial]),df_heartLabel)
    
    if testScore > ScoreBest:
        FSel=FTrial
        ScoreBest=testScore
    
        
print('Selected Subset : %s '%FSel)

print('Best Score of the Subset : %s '%ScoreBest)


# In[30]:

targetNoFeatures=7


for i in range(0,randomTrials):
    
    FTrial =random.sample(featureSet, k=targetNoFeatures)
    trainScore,testScore=featureSubsetScore(pd.DataFrame(heart_attributes[:, FTrial]),df_heartLabel)
    
    if testScore > ScoreBest:
        FSel=FTrial
        ScoreBest=testScore
    
        
print('Selected Subset : %s '%FSel)

print('Best Score of the Subset : %s '%ScoreBest)


# In[ ]:




# In[ ]:

def featureSubsetScore2(heartMatSub,Label):
    test_scores=list()
    train_scores=list()

    for seed in range(1, 11):
        X_train, X_test, y_train, y_test = train_test_split(heartMatSub, Label, 
                                                        test_size=0.2592, random_state=seed)
        
        # fit a model
        model = LogisticRegression(solver='lbfgs')
        model.fit(X_train, y_train)
        
        # predict labels on train set
        predTrain = model.predict(X_train)
        # evaluate the model
        scoreTrain = f1_score(y_train, predTrain)
        train_scores.append(scoreTrain)
        

        # predict labels on test set
        yhat = model.predict(X_test)
        # evaluate the model
        score = f1_score(y_test, yhat)
        test_scores.append(score)
        
        
    
    trainScoreMean=mean(train_scores)
    testScoreMean=mean(test_scores)
    
    return trainScoreMean,testScoreMean
    


# In[ ]:

trainScoreMean_old,testScoreMean_old=featureSubsetScore2(pd.DataFrame(heart_attributes[:, 0:12]),df_heartLabel)


# In[ ]:

trainScoreMean_old2,testScoreMean_old2=featureSubsetScore2(pd.DataFrame(heart_attributes[:, [0, 2,4]]),df_heartLabel)

