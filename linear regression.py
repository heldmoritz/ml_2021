import json
from tslearn.metrics import dtw, dtw_path
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.genmod.families import Poisson
from scipy import stats
import pandas as pd

with open('ae_train.json') as json_file:
    data_train = json.load(json_file)

with open('ae_test.json') as json_file:
    data_test = json.load(json_file)
    
#This function computes the features of each recording 
#The input is a datapoint which is an utterance of the vowel /ae/
def feature_function1(utterance):
    #define duration of the utterance
    length = len(utterance)
    
    #define number of features (for now 12 means)
    N = len(utterance[0])
    
    #define the empty output variable 
    features_per_utterance = np.zeros(N)
    
    #compute for each of the 12 coefficients the average value
    for n in np.arange(N):
        #define the empty feature vector
        features = np.zeros(length)
        
        #define the feature vector with length/dimension = duration of utterance
        for i in np.arange(length):
            features[i] = utterance[i][n]
        
        #compute the average of the n-th coefficient    
        features_per_utterance[n] = np.mean(features)   
    
    #return output feature vector
    return features_per_utterance

import random
#The input is a datapoint which is an utterance of the vowel /ae/
def feature_function2(utterance):
    #define duration of the utterance
    length = len(utterance)
    indices = range(length)
    
    weights = np.ones(length)
    weights[0] = 0.5
    weights[-1] = 0.5
    #define number of features per channel (7 as shortest recoring is 7)
    N = 7
    randoms = random.choices(indices, weights = weights, k=N)
       
    #define the empty feature vector
    features = []
    #compute each of the 7 features
    for n in randoms:
        features.append(np.mean(utterance[n]))
        features.append(np.var(utterance[n]))
        #take each value for each of the 12 channels
        for k in np.arange(12):
            features.append(utterance[n][k])
    
    #return output feature vector
    return features

#The next code compute the teacher varables y and matrix X for train and test data:
X_train = []
y_train = []
X_test = []
y_test = []

#sort the data into a numpy arrays
for speaker in data_train:
    for recording in data_train[speaker]:
        #compute the average coefficient values for each datapoint
        X_train.append(np.array(feature_function2(data_train[speaker][recording])))
        y_train.append(list(data_train).index(speaker))

for speaker in data_test:
    for recording in data_test[speaker]:
        #compute the average coefficient values for each datapoint
        X_test.append(np.array(feature_function2(data_test[speaker][recording])))
        y_test.append(list(data_test).index(speaker))

#add the 1 at the front for the intercept
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)  
#print(y_test)

#The next code is to do the linear regression on the training data    
fam = sm.families.NegativeBinomial()
p_model = sm.RLM(y_train, X_train, family=fam, M=sm.robust.norms.HuberT())
result = p_model.fit()
print(result.summary())
#print(result.params)

#Use in built function to predict test data
coefs = result.params
fits = p_model.predict(coefs,X_test)
#print(fits)
N = len(X_test)
round_fits = []
rights = 0
wrongs = 0
for j in np.arange(N):
    round_fits.append(round(fits[j]))
    if y_test[j]-round_fits[j] == 0:
        rights += 1
    else:
        wrongs += 1
    percentage_right = rights/N

print("In-built predicts for the test data:")    
print(rights)
print(wrongs)
print(percentage_right)
    





