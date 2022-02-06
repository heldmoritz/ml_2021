from dtaidistance import dtw_visualisation as dtwvis
from dtaidistance import dtw as dtwdist
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from tslearn.metrics import dtw, dtw_path

from util import get_x_y


plt.rcParams.update({"font.size": 18})


def plot_dtw_example(x_train):
    new_x_train = np.empty((270, 12), dtype=object)
    for i, sample in enumerate(x_train):
        for j, feature in enumerate(np.array(sample).T):
            new_x_train[i, j] = np.array(feature)

    timeseries1 = new_x_train[19, 1]  # 18,2
    timeseries2 = new_x_train[18, 1]  # 19,2

    d, my_paths = dtwdist.warping_paths(timeseries1, timeseries2, window=25, psi=2)
    best_path = dtwdist.best_path(my_paths)

    fig, ax = plt.subplots()
    plt.title('DTW distance between X and Y is %.2f' % d,)
    plt.ylabel('Amplitude')
    plt.xlabel('Time')
    plt.legend(['X', 'Y'])

    fig, ax = dtwvis.plot_warping_single_ax(timeseries1, timeseries2, path=best_path, fig=fig, ax=ax)
    ax.legend(['X', 'Y'])
    plt.show()
    # plt.savefig(fname='dtw_visualisation_singleax.svg')

    fig, ax = plt.subplots(2, 1)
    dtwvis.plot_warping(timeseries1, timeseries2, path=best_path, fig=fig, axs=ax,
                        filename='dtw_visualisation_twoax.pdf')

    plt.style.use('default')
    fig = plt.figure()
    dtwvis.plot_warpingpaths(timeseries1, timeseries2, paths=my_paths, path=best_path, showlegend=1, shownumbers=True)
    plt.show()
    best_path = dtwdist.best_path(my_paths)


def main():
    np.random.seed(123)
    max_n_neighbors = 10

    x_train, x_test, y_train, y_test = get_x_y()
    plot_dtw_example(x_train)

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
    clf = GridSearchCV(knn, param_grid={"n_neighbors": list(range(1, max_n_neighbors+1))}, cv=LeaveOneOut(), n_jobs=-1)
    clf.fit(train_precomputed, y_train)
    cv_results = clf.cv_results_
    plt.title("LOOCV on KNN with DTW\nfor different $K$ neighbors")
    plt.ylabel("Average accuracy")
    plt.xlabel("$K$ neighbors")
    plt.xticks(list(range(max_n_neighbors)), list(range(1, max_n_neighbors+1)))
    plt.plot(cv_results['mean_test_score'])
    plt.tight_layout()
    # plt.show()
    # print(clf.cv_results_)
    print(clf.best_params_)
    print(clf.best_score_)
    print(clf.score(test_precomputed, y_test))
    print("Accuracy per person")
    for label in set(y_test):
        x_label = [x for (x, y) in zip(test_precomputed, y_test) if y == label]
        score = clf.score(x_label, [label]*len(x_label))
        print(f"Person {label+1}: {int(score*len(x_label))}/{len(x_label)} ({score:.4f})")


if __name__ == '__main__':
    main()
