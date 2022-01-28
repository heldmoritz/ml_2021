import numpy as np
from hmmlearn import hmm

from tsf import get_x_y, shuffle_dependent_lists


def train_gmm_hmm(x_train, y_train, verbose=0) -> dict:
    gmm_hmm_models = {}

    for label in range(9):
        x_label = [x_train[i] for i in range(len(x_train)) if y_train[i] == label]
        if verbose > 0:
            print("Current person/label:", label)
            print(" Total recordings: ", len(x_label))
            print(" Recording lengths:", [len(x) for x in x_label])

        model = hmm.GaussianHMM()
        model.fit(np.vstack(x_label), lengths=[len(x) for x in x_label])

        gmm_hmm_models[label] = model
    
    return gmm_hmm_models


def hmm_score(x_train, y_train, x_test, y_test):
    """Trains HMM models on the training data and returns their accuracy on the test data.
    The accuracy is the fraction of test recordings which label matched the corresponding HMM model."""
    hmm_models = train_gmm_hmm(x_train, y_train)

    score_cnt = 0
    for i, recording in enumerate(x_test):
        best_score = -1
        best_label = None
        for model_label, model in hmm_models.items():
            score = model.score(recording)
            if score > best_score:
                best_score = score
                best_label = model_label
        if best_label == y_test[i]:
            score_cnt += 1

    return score_cnt/len(x_test)


def cross_validation(x_full, y_full, cv=5):
    """Performs k-fold cross validation on HMM models"""
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
    # Set a random seed for repeatable results
    np.random.seed(30)
    x_train, x_test, y_train, y_test = get_x_y()
    # print(cross_validation(x_train, y_train))
    print(hmm_score(x_train, y_train, x_test, y_test))


if __name__ == '__main__':
    main()
