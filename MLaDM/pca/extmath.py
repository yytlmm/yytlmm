import sys
import math

import numpy as np

def _fast_logdet(A):
    
    from scipy import linalg
    ld = np.sum(np.log(np.diag(A)))
    a = np.exp(ld/A.shape[0])
    d = np.linalg.det(A/a)
    ld += np.log(d)
    if not np.isfinite(ld):
        return -np.inf
    return ld

def _fast_logdet_numpy(A):
    
    from scipy import linalg
    sign, ld = np.linalg.slogdet(A)
    if not sign > 0:
        return -np.inf
    return ld


if hasattr(np.linalg, 'slogdet'):
    fast_logdet = _fast_logdet_numpy
else:
    fast_logdet = _fast_logdet

if sys.version_info[1] < 6:
   
    def factorial(x) :
      
        if x == 0: return 1
        return x * factorial(x-1)
else:
    factorial = math.factorial


if sys.version_info[1] < 6:
    def combinations(seq, r=None):
        
        if r == None:
            r = len(seq)
        if r <= 0:
            yield []
        else:
            for i in range(len(seq)):
                for cc in combinations(seq[i+1:], r-1):
                    yield [seq[i]]+cc

else:
    import itertools
    combinations = itertools.combinations


def density(w, **kwargs):
    
    d = 0 if w is None else float((w != 0).sum()) / w.size
    return d


def safe_sparse_dot(a, b, dense_output=False):
    
    from scipy import sparse
    if sparse.issparse(a) or sparse.issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a,b)


def fast_svd(M, k, p=None, q=0, transpose='auto', rng=0):
    
    from scipy import sparse

    if p == None:
        p = k

    if rng is None:
        rng = np.random.RandomState()
    elif isinstance(rng, int):
        rng = np.random.RandomState(rng)

    n_samples, n_features = M.shape

    if transpose == 'auto' and n_samples > n_features:
        transpose = True
    if transpose:
        
        M = M.T

  
    r = rng.normal(size=(M.shape[1], k + p))

    
    Y = safe_sparse_dot(M, r)
    del r

    for i in range(q):
        Y = safe_sparse_dot(M, safe_sparse_dot(M.T, Y))

    from .fixes import qr_economic
    Q, R = qr_economic(Y)
    del R

    B = safe_sparse_dot(Q.T, M)

    from scipy import linalg
    Uhat, s, V = linalg.svd(B, full_matrices=False)
    del B
    U = np.dot(Q, Uhat)

    if transpose:
        return V[:k, :].T, s[:k], U[:, :k].T
    else:
        return U[:, :k], s[:k], V[:k, :]

