import json

from hmmlearn import hmm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from util import shuffle_dependent_lists, get_x_y


DEFAULT_N_COMPONENTS = 1
DEFAULT_COVARIANCE_TYPE = "diag"
DEFAULT_MIN_COVAR = 1e-3
DEFAULT_ALGORITHM = "viterbi"
DEFAULT_N_ITER = 10
DEFAULT_TOL = 1e-2


plt.rcParams.update({'font.size': 18})


def train_gmm_hmm(x_train, y_train, params: dict) -> dict:
    """Trains n_classes HMM models."""
    hmm_models = {}

    for label in range(9):
        x_label = [x_train[i] for i in range(len(x_train)) if y_train[i] == label]
        model = hmm.GaussianHMM(**params)
        model.fit(np.vstack(x_label), lengths=[len(x) for x in x_label])
        hmm_models[label] = model
    
    return hmm_models


def hmm_score(x_train, y_train, x_test, y_test, params: dict):
    """Trains HMM models on the training data and returns their accuracy on the test data.
    The accuracy is the fraction of test recordings which label matched the corresponding HMM model."""
    hmm_models = train_gmm_hmm(x_train, y_train, params)

    score_cnt = 0
    for i, recording in enumerate(x_test):
        best_score = None
        best_label = None
        for model_label, model in hmm_models.items():
            score = model.score(recording)
            if best_score is None or score > best_score:
                best_score = score
                best_label = model_label
        if best_label == y_test[i]:
            score_cnt += 1

    return score_cnt/len(x_test)


def cv_chunks(x, y, k):
    raise NotImplementedError
    x, y, groups = indexable(x, y, None)
    cv = check_cv(k, y)
    for train, test in cv.split(x, y, groups):
        x_train, y_train = _safe_indexing(x, train), _safe_indexing(y, train)
        x_test, y_test = _safe_indexing(x, test), _safe_indexing(y, test)
        print(train, test)
    return
    div, rem = divmod(len(x_train)//n_classes, k)
    chunks = [[]]*k
    for label in range(9):
        x_label = [x for (i, x) in enumerate(x_train) if y_train[i] == label]
        i = 0
        rem_ = rem
        for chunk in chunks:
            for j in range(div):
                chunk.append(x_label[i])
                i += 1
            if rem_ > 0:
                chunk.append(x_label[i])
                i += 1
                rem_ -= 1
    return chunks


def cross_validation(x_full, y_full, params: dict, cv=5, verbose=0, shuffle=True):
    """Performs k-fold cross validation on HMM models"""
    if cv > len(x_full):
        raise ValueError(f"Not enough data (len(x_full)={len(x_full)}) to split into {cv} folds.")

    if verbose > 0:
        print(params)

    if shuffle:
        x_shuffle, y_shuffle = shuffle_dependent_lists(x_full, y_full)
        chunks = np.array_split(np.array(list(zip(x_shuffle, y_shuffle)), dtype=object), cv)
    else:
        chunks = np.array_split(np.array(list(zip(x_full, y_full)), dtype=object), cv)
    scores = []

    if verbose > 1:
        print("Progress: ", end="")

    for i in range(cv):
        x_test, y_test = zip(*chunks[i])
        x_train, y_train = [], []
        for j in range(cv):
            if j != i:
                x_temp, y_temp = zip(*chunks[j])
                x_train.extend(x_temp)
                y_train.extend(y_temp)

        scores.append(hmm_score(x_train, y_train, x_test, y_test, params=params))
        if verbose > 1:
            print(".", end="")
    if verbose > 1:
        print()
    return scores


def grid_search(x_train, y_train, n_iters=None, algorithms=None, tols=None, covariance_types=None, min_covars=None,
                n_components_list=None, verbose=0):
    raise NotImplementedError
    if n_iters is None:
        n_iters = [DEFAULT_N_ITER]
    if algorithms == "all":
        algorithms = ["viterbi", "map"]
    elif algorithms is None:
        algorithms = [DEFAULT_ALGORITHM]
    if tols is None:
        tols = [DEFAULT_TOL]
    if covariance_types == "all":
        covariance_types = ["diag", "spherical", "full", "tied"]
    elif covariance_types is None:
        covariance_types = [DEFAULT_COVARIANCE_TYPE]
    if min_covars is None:
        min_covars = [DEFAULT_MIN_COVAR]
    if n_components_list is None:
        n_components_list = [DEFAULT_N_COMPONENTS]

    best_config = {}
    best_score = -1
    for n_iter in n_iters:
        for algorithm in algorithms:
            for tol in tols:
                for covariance_type in covariance_types:
                    for min_covar in min_covars:
                        for n_components in n_components_list:
                            scores = cross_validation(x_train, y_train, n_iter=n_iter, algorithm=algorithm, tol=tol,
                                                      covariance_type=covariance_type, min_covar=min_covar,
                                                      n_components=n_components, verbose=verbose)
                            avg = np.mean(scores)
                            if verbose > 0:
                                print("Scores: ", end="")
                                print(", ".join([f"{x:.2f}" for x in scores]))
                                print(f"Average: {avg:.3f}")
                            if avg > best_score:
                                best_score = avg
                                best_config['n_iter'] = n_iter
                                best_config['algorithm'] = algorithm
                                best_config['tol'] = tol
                                best_config['covariance_type'] = covariance_type
                                best_config['min_covar'] = min_covar
                                best_config['n_components'] = n_components

    print("Best config:\n", best_config)
    print(f"Score: {best_score:.3f}")
    return best_config


def plot_accuracies(x_train, y_train):
    averages = []
    max_components = 15
    for n_components in range(1, max_components+1):
        scores = cross_validation(x_train, y_train, n_components=n_components, cv=15)
        averages.append(np.mean(scores))

    plt.title("Average $k$-fold accuracy for different $N$ components")
    plt.xlabel("$N$ components")
    plt.ylabel("Average accuracy")
    plt.xticks(list(range(max_components)), list(range(1, max_components+1)))
    plt.plot(averages)
    plt.show()


def pca_validate(x_train, y_train, x_test, y_test):
    x_train_pca = []
    for x in x_train:
        x_centered = StandardScaler().fit_transform(x)
        pca = PCA()
        pca.fit(x_centered)
        n_components = 0
        for eigenvector, eigenvalue in zip(pca.components_, pca.explained_variance_):
            if eigenvalue >= 1:
                n_components += 1
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x)
        x_train_pca.append(x_pca)
    config = grid_search(x_train_pca, y_train, n_components_list=[1, 2, 4, 8, 16, 32], verbose=0)
    x_test_pca = []
    for x in x_test:
        x_centered = StandardScaler().fit_transform(x)
        pca = PCA()
        pca.fit(x_centered)
        n_components = 0
        for eigenvector, eigenvalue in zip(pca.components_, pca.explained_variance_):
            if eigenvalue >= 1:
                n_components += 1
        pca = PCA(n_components=n_components)
        x_pca = pca.fit_transform(x)
        x_test_pca.append(x_pca)
    print(hmm_score(x_train_pca, y_train, x_test_pca, y_test, **config))


def leave_one_out(x_train, y_train, params: dict):
    scores = []
    for i in range(len(x_train)):
        x = [x_ for (j, x_) in enumerate(x_train) if j != i]
        y = [y_ for (j, y_) in enumerate(y_train) if j != i]
        x_val = [x_train[i]]
        y_val = [y_train[i]]
        scores.append(hmm_score(x, y, x_val, y_val, params))
    return scores


def print_confusion_matrix(x_train, y_train, x_test, y_test, n_components):
    hmm_models = train_gmm_hmm(x_train, y_train, {"n_components": n_components})
    confusion_matrix = np.zeros((9, 9), dtype=int)
    for recording, label in zip(x_test, y_test):
        best_score = None
        best_label = None
        for model_label, model in hmm_models.items():
            score = model.score(recording)
            if best_score is None or score > best_score:
                best_score = score
                best_label = model_label
        confusion_matrix[label][best_label] += 1

    cb_close = "}"
    cb_open = "{"
    for i, actual_label in enumerate(confusion_matrix):
        print(f"&\\textbf{cb_open}Sp.\\ {i+1}{cb_close} &", " & ".join(map(str, actual_label)), r"\\\cline{3-11}")


def plot_means(max_components=10, means=None):
    if means is None:
        max_components = 10
        with open("hmm_means.json", "r", encoding="utf-8") as f:
            means = json.load(f)
    plt.title("LOOCV on HMM\nfor different $N$ states")
    plt.xlabel("$N$ states")
    plt.ylabel("Average accuracy")
    plt.xticks(list(range(max_components)), list(range(1, max_components+1)))
    plt.plot(means)
    plt.tight_layout()
    plt.show()


def plot_recording_examples(x_train, y_train):
    recordings = []
    person = 0
    for x, y in zip(x_train, y_train):
        if person == 4:
            break
        if y == person:
            recordings.append(x)
            person += 1

    x_max = max(map(len, recordings))
    y_min = -1.5
    y_max = 2
    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(recordings[0])
    axs[0, 0].set_xlim([0, x_max])
    axs[0, 0].set_ylim([y_min, y_max])
    # axs[0, 0].set_title("Sp. 1, Rec. 1")
    axs[0, 1].plot(recordings[1])
    # axs[0, 1].set_title("Sp. 2, Rec. 1")
    axs[0, 1].set_xlim([0, x_max])
    axs[0, 1].set_ylim([y_min, y_max])
    axs[1, 0].plot(recordings[2])
    # axs[1, 0].set_title("Sp. 3, Rec. 1")
    axs[1, 0].set_xlim([0, x_max])
    axs[1, 0].set_ylim([y_min, y_max])
    axs[1, 1].plot(recordings[3])
    # axs[1, 1].set_title("Sp. 4, Rec. 1")
    axs[1, 1].set_xlim([0, x_max])
    axs[1, 1].set_ylim([y_min, y_max])

    for ax in axs.flat:
        ax.set(xlabel="Time", ylabel="Energy")

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.suptitle("Cepstrum coefficients over time\nfor the first recording\nof the first four speakers")
    plt.show()


def main():
    # Set a random seed for repeatable results
    np.random.seed(123)

    max_components = 10

    x_train, x_test, y_train, y_test = get_x_y()

    # plot_accuracies(x_train, y_train)
    means = []
    best_n_components = None
    best_score = None
    for n_components in range(1, max_components+1):
        print(f"{n_components:>2} components: ")
        scores = leave_one_out(x_train, y_train, {"n_components": n_components})
        mean = np.mean(scores)
        means.append(mean)
        print(f"    Average: {mean:.3f}")
        if best_score is None or mean > best_score:
            best_score = mean
            best_n_components = n_components
    with open("hmm_means.json", "w", encoding="utf-8") as f:
        json.dump(means, f)

    print(f"Best performing n_components: {best_n_components}")

    print(hmm_score(x_train, y_train, x_test, y_test, {"n_components": best_n_components}))
    plt.title("LOOCV on HMM\nfor different $N$ states")
    plt.xlabel("$N$ states")
    plt.ylabel("Average accuracy")
    plt.xticks(list(range(max_components)), list(range(1, max_components+1)))
    plt.plot(means)
    plt.tight_layout()
    plt.show()
    print_confusion_matrix(x_train, y_train, x_test, y_test, best_n_components)


if __name__ == '__main__':
    main()
