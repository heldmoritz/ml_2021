import json
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

np.random.seed(30)

with open('ae_train.json') as json_file:
    data_train = json.load(json_file)

with open('ae_test.json') as json_file:
    data_test = json.load(json_file)

#The next code compute the teacher varables y and matrix X for train and test data:
x_train = []
y_train = []
x_test = []
y_test = []

#sort the data into a numpy arrays
for speaker in data_train:
    for recording in data_train[speaker]:
        x_train.append(np.array(data_train[speaker][recording]))
        y_train.append(speaker)

for speaker in data_test:
    for recording in data_test[speaker]:
        x_test.append(np.array(data_test[speaker][recording]))
        y_test.append(speaker)


def train_GMMhmm(dataset):
    GMMHMM_Models = {}
    #The number of states
    states_num = 9
    
    #The number of ?
    GMM_mix_num = 9
    
    #Define transition probabilities in matrix form
    tmp_p = 1.0/(states_num-6)
    transmatPrior = np.array([[tmp_p, tmp_p, tmp_p, 0 ,0, 0, 0, 0, 0], \
                               [0, tmp_p, tmp_p, tmp_p , 0, 0, 0, 0, 0], \
                               [0, 0, tmp_p, tmp_p,tmp_p, 0, 0, 0, 0], \
                               [0, 0, 0, tmp_p,tmp_p, tmp_p, 0, 0, 0], \
                               [0, 0, 0, 0, tmp_p, tmp_p,tmp_p, 0, 0], \
                               [0, 0, 0, 0, 0, tmp_p, tmp_p,tmp_p, 0], \
                               [0, 0, 0, 0, 0, 0, tmp_p, tmp_p,tmp_p], \
                               [0, 0, 0, 0, 0, 0, 0, 0.5, 0.5], \
                               [0.5, 0, 0, 0, 0, 0, 0, 0, 0.5]],dtype=np.float)

    #Define the starting probabilities
    startprobPrior = np.array([0.2, 0.2, 0.3, 0.3, 0, 0, 0, 0, 0],dtype=np.float)

    for speaker in dataset:
        
        model = hmm.GMMHMM(n_components=states_num, n_mix=GMM_mix_num, \
                           transmat_prior=transmatPrior, startprob_prior=startprobPrior, \
                           covariance_type='diag', n_iter=10)
        
        length = np.ones(len(dataset[speaker]))
        
        for i in np.arange(len(dataset[speaker])):       
            length[i] = len(dataset[speaker][recording])
        
        trainData = np.vstack(dataset[speaker])
        
        #get the optimal parameters
        model.fit(trainData, lengths=length)
        
        GMMHMM_Models[speaker] = model
    
    return GMMHMM_Models

def main():
    trainDataSet = data_train
    print("Finish prepare the training data")

    hmmModels = train_GMMhmm(trainDataSet)
    print("Finish training of the GMM_HMM models for digits 0-9")

    testDataSet = data_test

    score_cnt = 0
    for label in testDataSet.keys():
        feature = testDataSet[label]
        scoreList = {}
        for model_label in hmmModels.keys():
            model = hmmModels[model_label]
            score = model.score(feature[0])
            scoreList[model_label] = score
        predict = max(scoreList, key=scoreList.get)
        
        print("Test on true label ", label, ": predict result label is ", predict)
        if predict == label:
            score_cnt+=1
    print("Final recognition rate is %.2f"%(100.0*score_cnt/len(testDataSet.keys())), "%")


if __name__ == '__main__':
    main()

