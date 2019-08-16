import warnings

import numpy as np

from found import BaseEstimator
from private import euclidean_distances


def k_init(X, k, n_samples_max=500, rng=None):
    
    n_samples = X.shape[0]
    if rng is None:
        rng = np.random

    if n_samples >= n_samples_max:
        X = X[rng.randint(n_samples, size=n_samples_max)]
        n_samples = n_samples_max

    distances = euclidean_distances(X, X, squared=True)

    # choose the 1st seed randomly, and store D(x)^2 in D[]
    first_idx = rng.randint(n_samples)
    centers = [X[first_idx]]
    D = distances[first_idx]

    for _ in range(k - 1):
        best_d_sum = best_idx = -1

        for i in range(n_samples):
            # d_sum = sum_{x in X} min(D(x)^2, ||x - xi||^2)
            d_sum = np.minimum(D, distances[i]).sum()

            if best_d_sum < 0 or d_sum < best_d_sum:
                best_d_sum, best_idx = d_sum, i

        centers.append(X[best_idx])
        D = np.minimum(D, distances[best_idx])

    return np.array(centers)


def k_means(X, k, init='k-means++', n_init=10, max_iter=300, verbose=0,
                    tol=1e-4, rng=None, copy_x=True):
    
    if rng is None:
        rng = np.random
    n_samples = X.shape[0]

    vdata = np.mean(np.var(X, 0))
    best_inertia = np.infty
    if hasattr(init, '__array__'):
        init = np.asarray(init)
        if not n_init == 1:
            warnings.warn('Explicit initial center position passed: '
                          'performing only one init in the k-means')
            n_init = 1
    'subtract of mean of x for more accurate distance computations'
    Xmean = X.mean(axis=0)
    if copy_x:
        X = X.copy()
    X -= Xmean
    for it in range(n_init):
        # init
        if init == 'k-means++':
            centers = k_init(X, k, rng=rng)
        elif init == 'random':
            seeds = np.argsort(rng.rand(n_samples))[:k]
            centers = X[seeds]
        elif hasattr(init, '__array__'):
            centers = np.asanyarray(init).copy()
        elif callable(init):
            centers = init(X, k, rng=rng)
        else:
            raise ValueError("the init parameter for the k-means should "
                "be 'k-means++' or 'random' or an ndarray, "
                "'%s' (type '%s') was passed.")

        if verbose:
            print('Initialization complete')
        # iterations
        x_squared_norms = X.copy()
        x_squared_norms **=2
        x_squared_norms = x_squared_norms.sum(axis=1)
        for i in range(max_iter):
            centers_old = centers.copy()
            labels, inertia = _e_step(X, centers,
                                        x_squared_norms=x_squared_norms)
            centers = _m_step(X, labels, k)
            if verbose:
                print('Iteration %i, inertia %s' % (i, inertia))
            if np.sum((centers_old - centers) ** 2) < tol * vdata:
                if verbose:
                    print('Converged to similar centers at iteration', i)
                break

            if inertia < best_inertia:
                best_labels = labels.copy()
                best_centers = centers.copy()
                best_inertia = inertia
    else:
        best_labels = labels
        best_centers = centers
        best_inertia = inertia
    if not copy_x:
        X += Xmean
    return best_centers + Xmean, best_labels, best_inertia


def _m_step(x, z, k):
    
    dim = x.shape[1]
    centers = np.empty((k, dim))
    X_center = None
    for q in range(k):
        this_center_mask = (z == q)
        if not np.any(this_center_mask):
            
            if X_center is None:
                X_center = x.mean(axis=0)
            centers[q] = X_center
        else:
            centers[q] = np.mean(x[this_center_mask], axis=0)
    return centers


def _e_step(x, centers, precompute_distances=True, x_squared_norms=None):
    
    n_samples = x.shape[0]
    k = centers.shape[0]

    if precompute_distances:
        distances = euclidean_distances(centers, x, x_squared_norms,
                                        squared=True)
    z = np.empty(n_samples, dtype=np.int)
    z.fill(-1)
    mindist = np.empty(n_samples)
    mindist.fill(np.infty)
    for q in range(k):
        if precompute_distances:
            dist = distances[q]
        else:
            dist = np.sum((x - centers[q]) ** 2, axis=1)
        z[dist < mindist] = q
        mindist = np.minimum(dist, mindist)
    inertia = mindist.sum()
    return z, inertia


class KMeans(BaseEstimator):

    def __init__(self, k=8, init='random', n_init=10, max_iter=300, tol=1e-4,
            verbose=0, rng=None, copy_x=True):
        self.k = k
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.verbose = verbose
        self.rng = rng
        self.copy_x = copy_x

    def fit(self, X, **params):
        X = np.asanyarray(X)
        self._set_params(**params)
        self.cluster_centers_, self.labels_, self.inertia_ = k_means(
            X, k=self.k, init=self.init, n_init=self.n_init,
            max_iter=self.max_iter, verbose=self.verbose,
            tol=self.tol, rng=self.rng, copy_x=self.copy_x)
        return self

