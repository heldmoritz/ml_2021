import json
import random
import typing

from tsf import build_x_y


def shuffle_dependent_lists(*lists: typing.Iterable):
    """Shuffle multiple lists, but keep the dependency between them"""
    tmp = list(zip(*lists))
    # Seed the random generator so results are consistent between runs
    random.Random(123).shuffle(tmp)
    return zip(*tmp)


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