
import numpy as np

from base import BaseEstimator
import cluster


def logsum(A, axis=None):
   
    Amax = A.max(axis)
    if axis and A.ndim > 1:
        shape = list(A.shape)
        shape[axis] = 1
        Amax.shape = shape
    Asum = np.log(np.sum(np.exp(A - Amax), axis))
    Asum += Amax.reshape(Asum.shape)
    if axis:
        
        Asum[np.isnan(Asum)] = - np.Inf
    return Asum



def normalize(A, axis=None):
    A += np.finfo(float).eps
    Asum = A.sum(axis)
    if axis and A.ndim > 1:
        
        Asum[Asum == 0] = 1
        shape = list(A.shape)
        shape[axis] = 1
        Asum.shape = shape
    return A / Asum


def lmvnpdf(obs, means, covars, cvtype='diag'):
    
    lmvnpdf_dict = {'spherical': _lmvnpdfspherical,
                    'tied': _lmvnpdftied,
                    'diag': _lmvnpdfdiag,
                    'full': _lmvnpdffull}
    return lmvnpdf_dict[cvtype](obs, means, covars)


def sample_gaussian(mean, covar, cvtype='diag', n_samples=1):
   
    n_dim = len(mean)
    rand = np.random.randn(n_dim, n_samples)
    if n_samples == 1:
        rand.shape = (n_dim,)

    if cvtype == 'spherical':
        rand *= np.sqrt(covar)
    elif cvtype == 'diag':
        rand = np.dot(np.diag(np.sqrt(covar)), rand)
    else:
        from scipy import linalg
        U, s, V = linalg.svd(covar)
        sqrtS = np.diag(np.sqrt(s))
        sqrt_covar = np.dot(U, np.dot(sqrtS, V))
        rand = np.dot(sqrt_covar, rand)

    return (rand.T + mean).T


class GMM(BaseEstimator): 

    def __init__(self, n_states=1, cvtype='diag'):
        self._n_states = n_states
        self._cvtype = cvtype

        if not cvtype in ['spherical', 'tied', 'diag', 'full']:
            raise ValueError('bad cvtype')

        self.weights = np.ones(self._n_states) / self._n_states

    
    @property
    def cvtype(self):
       
        return self._cvtype

    @property
    def n_states(self):
       
        return self._n_states

    def _get_covars(self):
        
        if self.cvtype == 'full':
            return self._covars
        elif self.cvtype == 'diag':
            return [np.diag(cov) for cov in self._covars]
        elif self.cvtype == 'tied':
            return [self._covars] * self._n_states
        elif self.cvtype == 'spherical':
            return [np.eye(self.n_features) * f for f in self._covars]

    def _set_covars(self, covars):
        covars = np.asanyarray(covars)
        _validate_covars(covars, self._cvtype, self._n_states, self.n_features)
        self._covars = covars

    covars = property(_get_covars, _set_covars)

    def _get_means(self):
       
        return self._means

    def _set_means(self, means):
        means = np.asarray(means)
        if hasattr(self, 'n_features') and \
               means.shape != (self._n_states, self.n_features):
            raise ValueError('means must have shape (n_states, n_features)')
        self._means = means.copy()
        self.n_features = self._means.shape[1]

    means = property(_get_means, _set_means)

    def _get_weights(self):
       
        return np.exp(self._log_weights)

    def _set_weights(self, weights):
        if len(weights) != self._n_states:
            raise ValueError('weights must have length n_states')
        if not np.allclose(np.sum(weights), 1.0):
            raise ValueError('weights must sum to 1.0')

        self._log_weights = np.log(np.asarray(weights).copy())

    weights = property(_get_weights, _set_weights)

    def eval(self, obs):
       
        obs = np.asanyarray(obs)
        lpr = (lmvnpdf(obs, self._means, self._covars, self._cvtype)
               + self._log_weights)
        logprob = logsum(lpr, axis=1)
        posteriors = np.exp(lpr - logprob[:,np.newaxis])
        return logprob, posteriors

    def score(self, obs):
       
        logprob, posteriors = self.eval(obs)
        return logprob

    def decode(self, obs):
        
        logprob, posteriors = self.eval(obs)
        return logprob, posteriors.argmax(axis=1)

    def predict(self, X):
       
        logprob, components = self.decode(X)
        return components

    def predict_proba(self, X):
       
        logprob, posteriors = self.eval(X)
        return posteriors

    def rvs(self, n_samples=1):
        
        weight_pdf = self.weights
        weight_cdf = np.cumsum(weight_pdf)

        obs = np.empty((n_samples, self.n_features))
        rand = np.random.rand(n_samples)
        
        comps = weight_cdf.searchsorted(rand)
        
        for comp in xrange(self._n_states):
            
            comp_in_obs = (comp==comps)
            
            num_comp_in_obs = comp_in_obs.sum() 
            if num_comp_in_obs > 0:
                if self._cvtype == 'tied':
                    cv = self._covars
                else:
                    cv = self._covars[comp]
                obs[comp_in_obs] = sample_gaussian(
                    self._means[comp], cv, self._cvtype, num_comp_in_obs).T
        return obs

    def fit(self, X, n_iter=10, min_covar=1e-3, thresh=1e-2, params='wmc',
            init_params='wmc'):
       
        X = np.asanyarray(X)

        if hasattr(self, 'n_features') and self.n_features != X.shape[1]:
            raise ValueError('Unexpected number of dimensions, got %s but '
                             'expected %s' % (X.shape[1], self.n_features))

        self.n_features = X.shape[1]

        if 'm' in init_params:
            self._means = cluster.KMeans(
                k=self._n_states).fit(X).cluster_centers_
        elif not hasattr(self, 'means'):
                self._means = np.zeros((self.n_states, self.n_features))

        if 'w' in init_params or not hasattr(self, 'weights'):
            self.weights = np.tile(1.0 / self._n_states, self._n_states)

        if 'c' in init_params:
            cv = np.cov(X.T)
            if not cv.shape:
                cv.shape = (1, 1)
            self._covars = _distribute_covar_matrix_to_match_cvtype(
                cv, self._cvtype, self._n_states)
        elif not hasattr(self, 'covars'):
                self.covars = _distribute_covar_matrix_to_match_cvtype(
                    np.eye(self.n_features), self.cvtype, self.n_states)

        
        logprob = []
        for i in xrange(n_iter):
            
            curr_logprob, posteriors = self.eval(X)
            logprob.append(curr_logprob.sum())

            
            if i > 0 and abs(logprob[-1] - logprob[-2]) < thresh:
                break

           
            self._do_mstep(X, posteriors, params, min_covar)

        return self

    def _do_mstep(self, X, posteriors, params, min_covar=0):
            w = posteriors.sum(axis=0)
            avg_obs = np.dot(posteriors.T, X)
            norm = 1.0 / (w[:,np.newaxis] + 1e-200)

            if 'w' in params:
                self._log_weights = np.log(w / w.sum())
            if 'm' in params:
                self._means = avg_obs * norm
            if 'c' in params:
                covar_mstep_func = _covar_mstep_funcs[self._cvtype]
                self._covars = covar_mstep_func(self, X, posteriors,
                                                avg_obs, norm, min_covar)

            return w



def _lmvnpdfdiag(obs, means=0.0, covars=1.0):
    n_obs, n_dim = obs.shape

    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.sum(np.log(covars), 1)
                  + np.sum((means ** 2) / covars, 1)
                  - 2 * np.dot(obs, (means / covars).T)
                  + np.dot(obs ** 2, (1.0 / covars).T))
    return lpr


def _lmvnpdfspherical(obs, means=0.0, covars=1.0):
    cv = covars.copy()
    if covars.ndim == 1:
        cv = cv[:,np.newaxis]
    return _lmvnpdfdiag(obs, means, np.tile(cv, (1, obs.shape[-1])))


def _lmvnpdftied(obs, means, covars):
    from scipy import linalg
    n_obs, n_dim = obs.shape
    
    icv = linalg.pinv(covars)
    lpr = -0.5 * (n_dim * np.log(2 * np.pi) + np.log(linalg.det(covars))
                  + np.sum(obs * np.dot(obs, icv), 1)[:,np.newaxis]
                  - 2 * np.dot(np.dot(obs, icv), means.T)
                  + np.sum(means * np.dot(means, icv), 1))
    return lpr


def _lmvnpdffull(obs, means, covars):
    
    from scipy import linalg
    import itertools
    if hasattr(linalg, 'solve_triangular'):
        
        solve_triangular = linalg.solve_triangular
    else:
        
        solve_triangular = linalg.solve
    n_obs, n_dim = obs.shape
    nmix = len(means)
    log_prob = np.empty((n_obs,nmix))
    for c, (mu, cv) in enumerate(itertools.izip(means, covars)):
        cv_chol = linalg.cholesky(cv, lower=True)
        cv_log_det  = 2*np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol  = solve_triangular(cv_chol, (obs - mu).T, lower=True).T
        log_prob[:, c]  = -.5 * (np.sum(cv_sol**2, axis=1) + \
                           n_dim * np.log(2 * np.pi) + cv_log_det)

    return log_prob


def _validate_covars(covars, cvtype, nmix, n_dim):
    from scipy import linalg
    if cvtype == 'spherical':
        if len(covars) != nmix:
            raise ValueError("'spherical' covars must have length nmix")
        elif np.any(covars <= 0):
            raise ValueError("'spherical' covars must be non-negative")
    elif cvtype == 'tied':
        if covars.shape != (n_dim, n_dim):
            raise ValueError("'tied' covars must have shape (n_dim, n_dim)")
        elif (not np.allclose(covars, covars.T)
              or np.any(linalg.eigvalsh(covars) <= 0)):
            raise ValueError("'tied' covars must be symmetric, "
                             "positive-definite")
    elif cvtype == 'diag':
        if covars.shape != (nmix, n_dim):
            raise ValueError("'diag' covars must have shape (nmix, n_dim)")
        elif np.any(covars <= 0):
            raise ValueError("'diag' covars must be non-negative")
    elif cvtype == 'full':
        if covars.shape != (nmix, n_dim, n_dim):
            raise ValueError("'full' covars must have shape "
                             "(nmix, n_dim, n_dim)")
        for n,cv in enumerate(covars):
            if (not np.allclose(cv, cv.T)
                or np.any(linalg.eigvalsh(cv) <= 0)):
                raise ValueError("component %d of 'full' covars must be "
                                 "symmetric, positive-definite" % n)


def _distribute_covar_matrix_to_match_cvtype(tiedcv, cvtype, n_states):
    if cvtype == 'spherical':
        cv = np.tile(np.diag(tiedcv).mean(), n_states)
    elif cvtype == 'tied':
        cv = tiedcv
    elif cvtype == 'diag':
        cv = np.tile(np.diag(tiedcv), (n_states, 1))
    elif cvtype == 'full':
        cv = np.tile(tiedcv, (n_states, 1, 1))
    else:
        raise (ValueError,
               "cvtype must be one of 'spherical', 'tied', 'diag', 'full'")
    return cv


def _covar_mstep_diag(gmm, obs, posteriors, avg_obs, norm, min_covar):
    
    avg_obs2 = np.dot(posteriors.T, obs * obs) * norm
    avg_means2 = gmm._means ** 2
    avg_obs_means = gmm._means * avg_obs * norm
    return avg_obs2 - 2 * avg_obs_means + avg_means2 + min_covar


def _covar_mstep_spherical(*args):
    return _covar_mstep_diag(*args).mean(axis=1)


def _covar_mstep_full(gmm, obs, posteriors, avg_obs, norm, min_covar):
    
    cv = np.empty((gmm._n_states, gmm.n_features, gmm.n_features))
    for c in xrange(gmm._n_states):
        post = posteriors[:,c]
        avg_cv = np.dot(post * obs.T, obs) / post.sum()
        mu = gmm._means[c][np.newaxis]
        cv[c] = (avg_cv - np.dot(mu.T, mu)
                 + min_covar * np.eye(gmm.n_features))
    return cv


def _covar_mstep_tied2(*args):
    return _covar_mstep_full(*args).mean(axis=0)


def _covar_mstep_tied(gmm, obs, posteriors, avg_obs, norm, min_covar):
    print "THIS IS BROKEN"
    
    avg_obs2 = np.dot(obs.T, obs)
    avg_means2 = np.dot(gmm._means.T, gmm._means)
    return (avg_obs2 - avg_means2 + min_covar * np.eye(gmm.n_features))


def _covar_mstep_slow(gmm, obs, posteriors, avg_obs, norm, min_covar):
    w = posteriors.sum(axis=0)
    covars = np.zeros(gmm._covars.shape)
    for c in xrange(gmm._n_states):
        mu = gmm._means[c]
        
        avg_obs2 = np.zeros((gmm.n_features, gmm.n_features))
        for t,o in enumerate(obs):
            avg_obs2 += posteriors[t,c] * np.outer(o, o)
        cv = (avg_obs2 / w[c]
              - 2 * np.outer(avg_obs[c] / w[c], mu)
              + np.outer(mu, mu)
              + min_covar * np.eye(gmm.n_features))
        if gmm.cvtype == 'spherical':
            covars[c] = np.diag(cv).mean()
        elif gmm.cvtype == 'diag':
            covars[c] = np.diag(cv)
        elif gmm.cvtype == 'full':
            covars[c] = cv
        elif gmm.cvtype == 'tied':
            covars += cv / gmm._n_states
    return covars


_covar_mstep_funcs = {'spherical': _covar_mstep_spherical,
                      'diag': _covar_mstep_diag,
                      #'tied': _covar_mstep_tied,
                      'full': _covar_mstep_full,
                      'tied': _covar_mstep_slow,
                      }
