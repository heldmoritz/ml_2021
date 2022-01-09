import json
import random

import numpy as np
from pyts.classification import TimeSeriesForest
# from pyts.datasets import load_gunpoint
from scipy.stats import linregress
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import typing


def series2features(time_series: typing.List):
    # start_pos = random.randint(0, len(time_series)-1)
    # end_pos = random.randint(start_pos, len(time_series))
    # interval_series = time_series[start_pos:end_pos]
    # return [np.mean(interval_series), np.std(interval_series), linregress(interval_series).slope]
    return [np.mean(time_series), np.std(time_series), linregress(list(range(0, len(time_series))), time_series).slope]


def shuffle_dependent_lists(*lists):
    """Shuffle two lists, but keep the dependency between them"""
    tmp = list(zip(*lists))
    # Seed the random generator so results are consistent between runs
    random.Random(123).shuffle(tmp)
    return zip(*tmp)


def get_x_y() -> tuple:
    with open("ae_train.json", "r") as f:
        data = json.load(f)  # type: dict
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    min_length = 2**32-1
    for i, (person, recordings) in enumerate(data.items()):
        for _, recording in recordings.items():
            min_length = min(min_length, len(recording))
            # x_train.append(series2features(list(map(np.mean, recording))))
            x_train.append(list(map(np.mean, recording)))
            y_train.append(i)
    with open("ae_test.json", "r") as f:
        data = json.load(f)  # type: dict
    for i, (person, recordings) in enumerate(data.items()):
        for _, recording in recordings.items():
            min_length = min(min_length, len(recording))
            # x_test.append(series2features(list(map(np.mean, recording))))
            x_test.append(list(map(np.mean, recording)))
            y_test.append(i)

    return [x[:min_length] for x in x_train], [x[:min_length] for x in x_test], y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_x_y()
    clf = TimeSeriesForest(random_state=43, n_jobs=-1)
    X_train, y_train = shuffle_dependent_lists(X_train, y_train)
    scores = cross_val_score(clf, X_train, y_train, cv=8, n_jobs=-1)
    print(scores, np.mean(scores))
    # clf.fit(X_train, y_train)
    # y_predict = clf.predict(X_test)
    # print(accuracy_score(y_test, y_predict))
    # print(clf.score(X_test, y_test))
