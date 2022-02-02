import json
import numpy as np
import random
import typing
from hmmlearn import hmm

#from sklearn.decomposition import PCA

def shuffle_dependent_lists(*lists: typing.Iterable):
    """Shuffle multiple lists, but keep the dependency between them"""
    tmp = list(zip(*lists))
    # Seed the random generator so results are consistent between runs
    random.Random(123).shuffle(tmp)
    return zip(*tmp)

def build_x_y(items, n_components=2):
    x = []
    y = []
    
    for i, (person, recordings) in enumerate(items):
        for _, recording in recordings.items():
            x.append(recording)
            y.append(i)

    x = np.array(x, dtype=object)
    y = np.array(y, dtype=int)
    return x, y

#This function is used to extra the data into seperate variables
def get_x_y(trim=False, n_components=2):
    with open("ae_train.json", "r") as f:
        data = json.load(f)  # type: dict
    x_train, y_train = build_x_y(data.items(), n_components=n_components)

    with open("ae_test.json", "r") as f:
        data = json.load(f)  # type: dict
    x_test, y_test = build_x_y(data.items(), n_components=n_components)

    return x_train, x_test, y_train, y_test


#This function is used to train a HMM for each speaker
def train_hmm(x_train, y_train):
    GaussianHMM_Models = {}
    #The number of states for each speaker
    N_states = 9
          
    for label in range(9):
        x_label = [x_train[i] for i in range(len(x_train)) if y_train[i] == label]
        train_data = np.vstack(x_label)
        
        #Train the HMM model
        model = hmm.GaussianHMM(n_components=N_states, covariance_type="diag", n_iter=20)
        
        # get the optimal parameters
        model.fit(train_data, lengths=[len(x) for x in x_label])

        GaussianHMM_Models[label] = model
        
        #The output is a sequence of quantities per speaker
    return GaussianHMM_Models

#This function computes the number of correct classifications for new data
def hmm_score(x_train, y_train, x_test, y_test):
    #Train all 9 HMM models for all speakers
    hmm_models = train_hmm(x_train, y_train)

    score_cnt = 0
    for label in range(9):
        recordings = [x_test[i] for i in range(len(x_test)) if y_test[i] == label]
        
        for recording in recordings:
            #Initialize score
            best_score = -1
            best_label = None
            for model_label, model in hmm_models.items():
                #Compute the score which measures the log probability that the recording matches with the compared model of a certain speaker
                score = model.score(recording)
                if score > best_score:
                    best_score = score
                    best_label = model_label
            if best_label == label:
                score_cnt += 1

    return score_cnt/len(x_test)

def cross_validation(x_full, y_full, cv=5):
    if cv > len(x_full):
        raise ValueError(f"Not enough data (len(x_full)={len(x_full)}) to split into {cv} folds.")
    
    #Shuffle the data around
    x_shuffle, y_shuffle = shuffle_dependent_lists(x_full, y_full)
    chunks = np.array_split(list(zip(x_shuffle, y_shuffle)), cv)
    scores = []
    for i in range(cv):
        x_test, y_test = zip(*chunks[i])
        x_train, y_train = [], []
        for j in range(cv):
            if j != i:
                x_temp, y_temp = zip(*chunks[j])
                x_train.extend(x_temp)
                y_train.extend(y_temp)

        scores.append(hmm_score(x_train, y_train, x_test, y_test))
    return np.array(scores,dtype=float)

def main():
    np.random.seed(30)
    x_train, x_test, y_train, y_test = get_x_y()
    print("Validation accuracy:")
    print(cross_validation(x_train, y_train))
    print("Test data accuracy:")
    print(hmm_score(x_train, y_train, x_test, y_test))


if __name__ == '__main__':
    main()

