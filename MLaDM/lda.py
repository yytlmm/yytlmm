
import warnings

import numpy as np
from scipy import linalg, ndimage

from base import BaseEstimator, ClassifierMixin


class LDA(BaseEstimator, ClassifierMixin):
    

    def __init__(self, priors=None):
        self.priors = np.asarray(priors) if priors is not None else None

    def fit(self, X, y, store_covariance=False, tol=1.0e-4, **params):
      
        self._set_params(**params)
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        if y.dtype.char.lower() not in ('b', 'h', 'i'):
            
            y = y.astype(np.int32)
        if X.ndim != 2:
            raise ValueError('X must be a 2D array')
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                'Incompatible shapes: X has %s samples, while y '
                'has %s' % (X.shape[0], y.shape[0]))
        n_samples = X.shape[0]
        n_features = X.shape[1]
        classes = np.unique(y)
        n_classes = classes.size
        if n_classes < 2:
            raise ValueError('y has less than 2 classes')
        classes_indices = [(y == c).ravel() for c in classes]
        if self.priors is None:
            counts = np.array(ndimage.measurements.sum(
                np.ones(n_samples, dtype=y.dtype), y, index=classes))
            self.priors_ = counts / float(n_samples)
        else:
            self.priors_ = self.priors

        # Group means n_classes*n_features matrix
        means = []
        Xc = []
        cov = None
        if store_covariance:
            cov = np.zeros((n_features, n_features))
        for group_indices in classes_indices:
            Xg = X[group_indices, :]
            meang = Xg.mean(0)
            means.append(meang)
            # centered group data
            Xgc = Xg - meang
            Xc.append(Xgc)
            if store_covariance:
                cov += np.dot(Xgc.T, Xgc)
        if store_covariance:
            cov /= (n_samples - n_classes)
            self.covariance_ = cov

        means = np.asarray(means)
        Xc = np.concatenate(Xc, 0)

       
        scaling = 1. / Xc.std(0)
        fac = float(1) / (n_samples - n_classes)
        
        X = np.sqrt(fac) * (Xc * scaling)
       
        U, S, V = linalg.svd(X, full_matrices=0)

        rank = np.sum(S > tol)
        if rank < n_features:
            warnings.warn("Variables are collinear")
       
        scaling = (scaling * V.T[:, :rank].T).T / S[:rank]
        
        xbar = np.dot(self.priors_, means)
       
        X = np.dot(((np.sqrt((n_samples * self.priors_)*fac)) *
                          (means - xbar).T).T, scaling)
       
        _, S, V = linalg.svd(X, full_matrices=0)

        rank = np.sum(S > tol*S[0])
        
        scaling = np.dot(scaling, V.T[:, :rank])
        self.scaling = scaling
        self.means_ = means
        self.xbar_ = xbar
        self.classes = classes
        return self

    def decision_function(self, X):
       
        X = np.asanyarray(X)
        scaling = self.scaling
        
        X = np.dot(X - self.xbar_, scaling)
        
        dm = np.dot(self.means_ - self.xbar_, scaling)
        
        return -0.5 * np.sum(dm ** 2, 1) + \
                np.log(self.priors_) + np.dot(X, dm.T)

    def predict(self, X):
       
        d = self.decision_function(X)
        y_pred = self.classes[d.argmax(1)]
        return y_pred

    def predict_proba(self, X):
      
        values = self.decision_function(X)
       
        likelihood = np.exp(values - values.min(axis=1)[:, np.newaxis])
        
        return likelihood / likelihood.sum(axis=1)[:, np.newaxis]

    def predict_log_proba(self, X):
       
        probas_ = self.predict_proba(X)
        return np.log(probas_)
