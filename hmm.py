import numpy as np
from hmmlearn import hmm

from util import shuffle_dependent_lists, get_x_y


DEFAULT_N_COMPONENTS = 1
DEFAULT_COVARIANCE_TYPE = "diag"
DEFAULT_MIN_COVAR = 1e-3
DEFAULT_ALGORITHM = "viterbi"
DEFAULT_N_ITER = 10
DEFAULT_TOL = 1e-2


def train_gmm_hmm(x_train, y_train, n_iter=None, algorithm=None, tol=None, covariance_type=None, min_covar=None,
                  n_components=None) -> dict:
    """Trains n_classes HMM models."""
    kwargs = {}

    if n_iter is not None:
        kwargs['n_iter'] = n_iter
    if algorithm is not None:
        kwargs['algorithm'] = algorithm
    if tol is not None:
        kwargs['tol'] = tol
    if covariance_type is not None:
        kwargs['covariance_type'] = covariance_type
    if min_covar is not None:
        kwargs['min_covar'] = min_covar
    if n_components is not None:
        kwargs['n_components'] = n_components

    hmm_models = {}

    for label in range(9):
        x_label = [x_train[i] for i in range(len(x_train)) if y_train[i] == label]

        model = hmm.GaussianHMM(**kwargs)
        model.fit(np.vstack(x_label), lengths=[len(x) for x in x_label])

        hmm_models[label] = model
    
    return hmm_models


def hmm_score(x_train, y_train, x_test, y_test, n_iter=None, algorithm=None, tol=None, covariance_type=None,
              min_covar=None, n_components=None):
    """Trains HMM models on the training data and returns their accuracy on the test data.
    The accuracy is the fraction of test recordings which label matched the corresponding HMM model."""
    hmm_models = train_gmm_hmm(x_train, y_train, n_iter=n_iter, algorithm=algorithm, tol=tol, min_covar=min_covar,
                               covariance_type=covariance_type, n_components=n_components)

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


def cross_validation(x_full, y_full, cv=5, n_iter=None, algorithm=None, tol=None, covariance_type=None, min_covar=None,
                     n_components=None, verbose=0):
    """Performs k-fold cross validation on HMM models"""
    if cv > len(x_full):
        raise ValueError(f"Not enough data (len(x_full)={len(x_full)}) to split into {cv} folds.")
    if n_iter is None:
        n_iter = DEFAULT_N_ITER
    if algorithm is None:
        algorithm = DEFAULT_ALGORITHM
    if tol is None:
        tol = DEFAULT_TOL
    if covariance_type is None:
        covariance_type = DEFAULT_COVARIANCE_TYPE
    if min_covar is None:
        min_covar = DEFAULT_MIN_COVAR
    if n_components is None:
        n_components = DEFAULT_N_COMPONENTS

    if verbose > 0:
        print(f"n_iter={n_iter}, algorithm={algorithm}, tol={tol}, covariance_type={covariance_type}, "
              f"min_covar={min_covar}, n_components={n_components}")

    x_shuffle, y_shuffle = shuffle_dependent_lists(x_full, y_full)
    chunks = np.array_split(np.array(list(zip(x_shuffle, y_shuffle)), dtype=object), cv)
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

        scores.append(hmm_score(x_train, y_train, x_test, y_test, n_iter=n_iter, algorithm=algorithm, tol=tol,
                                covariance_type=covariance_type, min_covar=min_covar, n_components=n_components))
        if verbose > 1:
            print(".", end="")
    print()
    return scores


def grid_search(x_train, y_train, n_iters=None, algorithms=None, tols=None, covariance_types=None, min_covars=None,
                n_components_list=None, verbose=0):
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


def main():
    # Set a random seed for repeatable results
    np.random.seed(30)
    x_train, x_test, y_train, y_test = get_x_y()
    config = grid_search(x_train, y_train, n_components_list=[1, 2, 4, 8, 16, 32], verbose=2, n_iters=[1, 2, 4],
                         algorithms="all", min_covars=[DEFAULT_MIN_COVAR, 0.1*DEFAULT_MIN_COVAR, 10*DEFAULT_MIN_COVAR],
                         tols=[DEFAULT_TOL, 0.1*DEFAULT_TOL, 10*DEFAULT_TOL])
    print(hmm_score(x_train, y_train, x_test, y_test, **config))


if __name__ == '__main__':
    main()
