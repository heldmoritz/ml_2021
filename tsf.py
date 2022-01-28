import json
import typing

import matplotlib.pyplot as plt
import numpy as np
from pyts.classification import TimeSeriesForest
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from util import shuffle_dependent_lists, get_x_y


def series2features(time_series: typing.List):
    # start_pos = random.randint(0, len(time_series)-1)
    # end_pos = random.randint(start_pos, len(time_series))
    # interval_series = time_series[start_pos:end_pos]
    # return [np.mean(interval_series), np.std(interval_series), linregress(interval_series).slope]
    return [np.mean(time_series), np.std(time_series), linregress(list(range(0, len(time_series))), time_series).slope]


def build_x_y(items, min_length, mean, trim, use_pca, n_components=2):
    x = []
    y = []
    shortest_recording = 2**32-1
    for i, (person, recordings) in enumerate(items):
        for _, recording in recordings.items():
            if len(recording) >= min_length:
                shortest_recording = min(shortest_recording, len(recording))
                if mean:
                    x.append(list(map(np.mean, recording)))
                else:
                    x.append(recording)
                y.append(i)
    if trim:
        x = [_x[:shortest_recording] for _x in x]
    if trim and use_pca:
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x)
        x = pca.inverse_transform(x_pca)
    else:
        x = np.array(x, dtype=object)
    y = np.array(y, dtype=int)
    return x, y


def test_pca():
    with open("ae_train.json", "r") as f:
        data = json.load(f)  # type: dict
    items = list(data.items())
    x = []
    y = []
    person, recordings = items[0]
    recording = recordings['recording0']
    pca = PCA(n_components=2)

    fig = plt.figure()
    plt.plot(recording)
    plt.show(block=False)

    recording_pca = pca.fit_transform(recording)
    fig2 = plt.figure()
    plt.plot(recording_pca)
    plt.show(block=False)
    inverse = pca.inverse_transform(recording_pca)
    fig3 = plt.figure()
    plt.plot(inverse)
    plt.show(block=False)
    fig4 = plt.figure()
    plt.plot(pca.components_[:2])
    plt.show(block=True)


def get_pca(x_train, n_components=2):
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    pca.inverse_transform(x_train_pca)


def grid_search(x, y, param_grid, classifier):
    gs = GridSearchCV(classifier, param_grid, cv=5)
    gs.fit(x, y)
    print(gs.best_params_)
    print(gs.best_score_)


def array2bins(arr, n_bins=10) -> typing.List:
    arr_binned = []
    for step in np.linspace(0, len(arr)-1, n_bins):
        if (step+1) >= len(arr):
            arr_binned.append(arr[-1])
        else:
            arr_binned.append(arr[int(step)] + (step % 1)*(arr[int(step)+1] - arr[int(step)]))
    return arr_binned


def test_bins(n_bins=10):
    print("\n", n_bins, "bins.")
    x_train, x_test, y_train, y_test = get_x_y(mean=True)
    x_train, y_train = shuffle_dependent_lists(x_train, y_train)

    x_train = [array2bins(x, n_bins=n_bins) for x in x_train]

    pg = {
        "n_windows": [1.0],
        "random_state": [43],
        "bootstrap": [True],
        "min_impurity_decrease": [0.0],
        "max_leaf_nodes": [25, 50, 75, 100, 125],
        "max_features": ["auto"],
        "criterion": ["entropy", "gini"],
        "min_samples_split": [2]
    }

    clf = TimeSeriesForest(n_jobs=1, verbose=0)
    grid_search(x_train, y_train, param_grid=pg, classifier=clf)


def perform_experiment(min_length=14, use_pca=False, trim=True, mean=True, n_components=1):
    print("\nStarting experiment...")
    if use_pca:
        print("\nUsing PCA with {n_components} components.")
    x_train, x_test, y_train, y_test = get_x_y(trim=trim, mean=mean, min_length=min_length, use_pca=use_pca,
                                               n_components=n_components)
    x_train, y_train = shuffle_dependent_lists(x_train, y_train)

    pg = {
        "n_windows": [1.0],
        "random_state": [43],
        "bootstrap": [True],
        "min_impurity_decrease": [0.0],
        "max_leaf_nodes": [25, 50, 75, 100, 125],
        "max_features": ["auto"],
        "criterion": ["entropy", "gini"],
        "min_samples_split": [2]
    }

    clf = TimeSeriesForest(n_jobs=1, verbose=0)
    grid_search(x_train, y_train, param_grid=pg, classifier=clf)


if __name__ == '__main__':
    # test_bins()
    # test_bins(8)
    # test_bins(12)
    # test_pca()
    perform_experiment()
    quit()
    for n in range(1, 6):
        perform_experiment(use_pca=True, n_components=n)

    # kernels = generate_kernels(np.shape(X_train)[-1], 10_000)
    #
    # # transform training set and train classifier
    # X_train_transform = apply_kernels(X_train, kernels)
    # clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
    # # pipeline = Pipeline([("transformer", Transformer(partial(apply_kernels, kernels=kernels))), ("clf", clf)])
    # # scores = cross_val_score(pipeline, X_train, y_train)
    # # print(scores)
    # clf.fit(X_train_transform, y_train)
    #
    # # transform test set and predict
    # X_test_transform = apply_kernels(X_test, kernels)
    # predictions = clf.predict(X_test_transform)
    # print(accuracy_score(y_test, predictions))
    #
    # pg = {
    #     "n_windows": [0.1, 0.5, 1.0],
    #     "random_state": [43],
    #     "bootstrap": [True, False],
    #     "min_impurity_decrease": [0.0, 0.05, 0.1]
    # }

    # scores = cross_val_score(grid_search, X_train, y_train, cv=5, n_jobs=-1)
    # print(grid_search.best_estimator_)
    # print(scores, np.mean(scores))
    # clf.fit(X_train, y_train)
    # y_predict = clf.predict(X_test)
    # print(accuracy_score(y_test, y_predict))
    # print(clf.score(X_test, y_test))
