import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from tslearn.metrics import dtw

from util import get_x_y


def main():
    x_train, x_test, y_train, y_test = get_x_y()
    train_precomputed = np.zeros((len(x_train), len(x_train)), dtype=float)
    for i, x in enumerate(x_train):
        for j, x2 in enumerate(x_train):
            if i != j:
                train_precomputed[i][j] = dtw(x, x2)

    test_precomputed = np.zeros((len(x_test), len(x_train)), dtype=float)
    for i, x in enumerate(x_test):
        for j, x2 in enumerate(x_train):
            test_precomputed[i][j] = dtw(x, x2)

    knn = KNeighborsClassifier(metric="precomputed")
    clf = GridSearchCV(knn, param_grid={"n_neighbors": list(range(1, 11))})
    clf.fit(train_precomputed, y_train)
    print(clf.cv_results_)
    print(clf.best_params_)
    print(clf.score(test_precomputed, y_test))


if __name__ == '__main__':
    main()
