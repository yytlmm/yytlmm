
import exceptions
import warnings

import numpy as np
import scipy.ndimage as ndimage

from base import BaseEstimator, ClassifierMixin


class QDA(BaseEstimator, ClassifierMixin):
    
    def __init__(self, priors=None):
        self.priors = np.asarray(priors) if priors is not None else None

    def fit(self, X, y, store_covariances=False, tol=1.0e-4, **params):
       
        self._set_params(**params)
        X = np.asanyarray(X)
        y = np.asanyarray(y)
        if X.ndim!=2:
            raise exceptions.ValueError('X must be a 2D array')
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                'Incompatible shapes: X has %s samples, while y '
                'has %s' % (X.shape[0], y.shape[0]))
        if y.dtype.char.lower() not in ('b', 'h', 'i'):
           
            y = y.astype(np.int32)
        n_samples, n_features = X.shape
        classes = np.unique(y)
        n_classes = classes.size
        if n_classes < 2:
            raise exceptions.ValueError('y has less than 2 classes')
        classes_indices = [(y == c).ravel() for c in classes]
        if self.priors is None:
            counts = np.array(ndimage.measurements.sum(
                np.ones(n_samples, dtype=y.dtype), y, index=classes))
            self.priors_ = counts / float(n_samples)
        else:
            self.priors_ = self.priors

        cov = None
        if store_covariances:
            cov = []
        means = []
        scalings = []
        rotations = []
        for group_indices in classes_indices:
            Xg = X[group_indices, :]
            meang = Xg.mean(0)
            means.append(meang)
            Xgc = Xg - meang
           
            U, S, Vt = np.linalg.svd(Xgc, full_matrices=False)
            rank = np.sum(S > tol)
            if rank < n_features:
                warnings.warn("Variables are collinear")
            S2 = (S ** 2) / (len(Xg) - 1)
            if store_covariances:
               
                cov.append(np.dot(S2 * Vt.T, Vt))
            scalings.append(S2)
            rotations.append(Vt.T)
        if store_covariances:
            self.covariances_ = cov
        self.means_ = np.asarray(means)
        self.scalings = np.asarray(scalings)
        self.rotations = rotations
        self.classes = classes
        return self

    def decision_function(self, X):
       
        X = np.asanyarray(X)
        norm2 = []
        for i in range(len(self.classes)):
            R = self.rotations[i]
            S = self.scalings[i]
            Xm = X - self.means_[i]
            X2 = np.dot(Xm, R * (S ** (-0.5)))
            norm2.append(np.sum(X2 ** 2, 1))
        norm2 = np.array(norm2).T # shape : len(X), n_classes
        return -0.5 * (norm2 + np.sum(np.log(self.scalings), 1)) + \
               np.log(self.priors_)

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
