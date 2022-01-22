import json
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

from tsf import get_x_y, shuffle_dependent_lists


def train_gmm_hmm(x_train, y_train):
    gmm_hmm_models = {}
    # The number of states
    states_num = 9
    
    # The number of ?
    mix_num = 9
    
    # Define transition probabilities in matrix form
    # tmp_p = 1.0/(states_num-6)
    # trans_mat_prior = np.array([[tmp_p, tmp_p, tmp_p, 0, 0, 0, 0, 0, 0],
    #                            [0, tmp_p, tmp_p, tmp_p, 0, 0, 0, 0, 0],
    #                            [0, 0, tmp_p, tmp_p, tmp_p, 0, 0, 0, 0],
    #                            [0, 0, 0, tmp_p, tmp_p, tmp_p, 0, 0, 0],
    #                            [0, 0, 0, 0, tmp_p, tmp_p, tmp_p, 0, 0],
    #                            [0, 0, 0, 0, 0, tmp_p, tmp_p, tmp_p, 0],
    #                            [0, 0, 0, 0, 0, 0, tmp_p, tmp_p, tmp_p],
    #                            [0, 0, 0, 0, 0, 0, 0, 0.5, 0.5],
    #                            [0.5, 0, 0, 0, 0, 0, 0, 0, 0.5]], dtype=float)

    # Define the starting probabilities
    # start_prob_prior = np.array([0.2, 0.2, 0.3, 0.3, 0, 0, 0, 0, 0], dtype=float)

    for label in range(9):
        x_label = [x_train[i] for i in range(len(x_train)) if y_train[i] == label]
        # print("Current person/label:", label)
        # print(" Total recordings: ", len(x_label))
        # print(" Recording lengths:", [len(x) for x in x_label])
        train_data = np.vstack(x_label)
        # for x in train_data:
        #     if len(x) != 12:
        #         print(f"Found {len(x)} channels instead of 12.")
        #     for y in x:
        #         if y == 0:
        #             print("Found a channel value of 0.")
        # with open(f"data_{label}.json", "w") as f:
        #     json.dump(train_data.tolist(), f)
        model = hmm.GaussianHMM(n_components=states_num, covariance_type="diag", n_iter=10)
        
        # get the optimal parameters
        model.fit(train_data, lengths=[len(x) for x in x_label])

        gmm_hmm_models[label] = model
    
    return gmm_hmm_models


def hmm_score(x_train, y_train, x_test, y_test):
    hmm_models = train_gmm_hmm(x_train, y_train)

    score_cnt = 0
    for label in range(9):
        recordings = [x_test[i] for i in range(len(x_test)) if y_test[i] == label]
        best_score = -1
        best_label = None
        for recording in recordings:
            for model_label, model in hmm_models.items():
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
    return scores


def main():
    np.random.seed(30)
    x_train, x_test, y_train, y_test = get_x_y()
    # print(cross_validation(x_train, y_train))
    print(hmm_score(x_train, y_train, x_test, y_test))


if __name__ == '__main__':
    main()
