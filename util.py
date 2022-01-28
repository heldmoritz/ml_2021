import json
import random
import typing

import numpy as np
from sklearn.decomposition import PCA


def shuffle_dependent_lists(*lists: typing.Iterable):
    """Shuffle multiple lists, but keep the dependency between them"""
    tmp = list(zip(*lists))
    # Seed the random generator so results are consistent between runs
    random.Random(123).shuffle(tmp)
    return zip(*tmp)


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


def get_x_y(trim=False, mean=False, min_length=0, use_pca=False, n_components=2):
    with open("ae_train.json", "r") as f:
        data = json.load(f)  # type: dict
    x_train, y_train = build_x_y(data.items(), min_length=min_length, mean=mean, trim=trim, use_pca=use_pca,
                                 n_components=n_components)

    with open("ae_test.json", "r") as f:
        data = json.load(f)  # type: dict
    x_test, y_test = build_x_y(data.items(), min_length=min_length, mean=mean, trim=trim, use_pca=use_pca,
                               n_components=n_components)

    return x_train, x_test, y_train, y_test
