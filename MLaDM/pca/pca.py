import numpy as np
from scipy import linalg

from under import BaseEstimator
from extmath import fast_logdet
from extmath import fast_svd
from extmath import safe_sparse_dot


def _assess_dimension_(spectrum, rank, n_samples, dim):
   
    if rank > dim:
        raise ValueError("the dimension cannot exceed dim")
    from scipy.special import gammaln

    pu = -rank * np.log(2)
    for i in range(rank):
        pu += gammaln((dim - i) / 2) - np.log(np.pi) * (dim - i) / 2

    pl = np.sum(np.log(spectrum[:rank]))
    pl = -pl * n_samples / 2

    if rank == dim:
        pv = 0
        v = 1
    else:
        v = np.sum(spectrum[rank:dim]) / (dim - rank)
        pv = -np.log(v) * n_samples * (dim - rank) / 2

    m = dim * rank - rank * (rank + 1) / 2
    pp = np.log(2 * np.pi) * (m + rank + 1) / 2

    pa = 0
    spectrum_ = spectrum.copy()
    spectrum_[rank:dim] = v
    for i in range(rank):
        for j in range(i + 1, dim):
            pa += (np.log((spectrum[i] - spectrum[j])
                          * (1. / spectrum_[j] - 1. / spectrum_[i]))
                   + np.log(n_samples))

    ll = pu + pl + pv + pp - pa / 2 - rank * np.log(n_samples) / 2

    return ll


def _infer_dimension_(spectrum, n, p):
   
    ll = []
    for rank in range(min(n, p, len(spectrum))):
        ll.append(_assess_dimension_(spectrum, rank, n, p))
    ll = np.array(ll)
    return ll.argmax()


class PCA(BaseEstimator):
   
    def __init__(self, n_components=None, copy=True, whiten=False):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten

    def fit(self, X, **params):
        
        self._set_params(**params)
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        if self.copy:
            X = X.copy()
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        U, S, V = linalg.svd(X, full_matrices=False)
        self.explained_variance_ = (S ** 2) / n_samples
        self.explained_variance_ratio_ = self.explained_variance_ / \
                                        self.explained_variance_.sum()

        if self.whiten:
            n = X.shape[0]
            self.components_ = np.dot(V.T, np.diag(1.0 / S)) * np.sqrt(n)
        else:
            self.components_ = V.T

        if self.n_components == 'mle':
            self.n_components = _infer_dimension_(self.explained_variance_,
                                            n_samples, X.shape[1])

        elif 0 < self.n_components and self.n_components < 1.0:
            
            n_remove = np.sum(self.explained_variance_ratio_.cumsum() >=
                              self.n_components) - 1
            self.n_components = n_features - n_remove

        if self.n_components is not None:
            self.components_ = self.components_[:, :self.n_components]
            self.explained_variance_ = \
                    self.explained_variance_[:self.n_components]
            self.explained_variance_ratio_ = \
                    self.explained_variance_ratio_[:self.n_components]

        return self

    def transform(self, X):
       
        X_transformed = X - self.mean_
        X_transformed = np.dot(X_transformed, self.components_)
        return X_transformed

    def inverse_transform(self, X):
       
        return np.dot(X, self.components_.T) + self.mean_


class ProbabilisticPCA(PCA):

    def fit(self, X, homoscedastic=True):
        
        PCA.fit(self, X)
        self.dim = X.shape[1]
        Xr = X - self.mean_
        Xr -= np.dot(np.dot(Xr, self.components_), self.components_.T)
        n_samples = X.shape[0]
        if self.dim <= self.n_components:
            delta = np.zeros(self.dim)
        elif homoscedastic:
            delta = (Xr ** 2).sum() * np.ones(self.dim) \
                    / (n_samples * self.dim)
        else:
            delta = (Xr ** 2).mean(0) / (self.dim - self.n_components)
        self.covariance_ = np.diag(delta)
        for k in range(self.n_components):
            add_cov = np.dot(
                self.components_[:, k:k + 1], self.components_[:, k:k + 1].T)
            self.covariance_ += self.explained_variance_[k] * add_cov
        return self

    def score(self, X):
      
        Xr = X - self.mean_
        log_like = np.zeros(X.shape[0])
        self.precision_ = np.linalg.inv(self.covariance_)
        for i in range(X.shape[0]):
            log_like[i] = -.5 * np.dot(np.dot(self.precision_, Xr[i]), Xr[i])
        log_like += fast_logdet(self.precision_) - \
                                    self.dim / 2 * np.log(2 * np.pi)
        return log_like


class RandomizedPCA(BaseEstimator):
    
    def __init__(self, n_components, copy=True, iterated_power=3,
                 whiten=False):
        self.n_components = n_components
        self.copy = copy
        self.iterated_power = iterated_power
        self.whiten = whiten
        self.mean_ = None

    def fit(self, X, **params):
        
        self._set_params(**params)
        n_samples = X.shape[0]

        if self.copy:
            X = X.copy()

        if not hasattr(X, 'todense'):
            
            X = np.atleast_2d(X)

            
            self.mean_ = np.mean(X, axis=0)
            X -= self.mean_

        U, S, V = fast_svd(X, self.n_components, q=self.iterated_power)

        self.explained_variance_ = (S ** 2) / n_samples
        self.explained_variance_ratio_ = self.explained_variance_ / \
                                        self.explained_variance_.sum()

        if self.whiten:
            n = X.shape[0]
            self.components_ = np.dot(V.T, np.diag(1.0 / S)) * np.sqrt(n)
        else:
            self.components_ = V.T

        return self

    def transform(self, X):
        
        if self.mean_ is not None:
            X = X - self.mean_

        X = safe_sparse_dot(X, self.components_)
        return X

    def inverse_transform(self, X):
        
        X_original = safe_sparse_dot(X, self.components_.T)
        if self.mean_ is not None:
            X_original = X_original + self.mean_
        return X_original

