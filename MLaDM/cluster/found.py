''' 
private something basics for k_means
'''
import inspect
import copy

import numpy as np


def r2_score(y_true, y_pred):
    return 1 - (((y_true - y_pred) ** 2).sum() /
                ((y_true - y_true.mean()) ** 2).sum())


################################################################################
def clone(estimator, safe=True):
  
    estimator_type = type(estimator)
    
    if estimator_type in (list, tuple, set, frozenset):
         return estimator_type([clone(e, safe=safe) for e in estimator])
    elif not hasattr(estimator, '_get_params'):
        if not safe:
            return copy.deepcopy(estimator)
        else:
            raise ValueError("Cannot clone object '%s' (type %s): "
                    "it does not seem to be a scikit-learn estimator as "
                    "it does not implement a '_get_params' methods."
                    % (repr(estimator), type(estimator)))
    klass = estimator.__class__
    new_object_params = estimator._get_params(deep=False)
    for name, param in new_object_params.iteritems():
        new_object_params[name] = clone(param, safe=False)
    new_object = klass(**new_object_params)
    assert new_object._get_params(deep=False) == new_object_params, (
            'Cannot clone object %s, as the constructor does not '
            'seem to set parameters' % estimator
        )

    return new_object


################################################################################
def _pprint(params, offset=0, printer=repr):
   
    options = np.get_printoptions()
    np.set_printoptions(precision=5, threshold=64, edgeitems=2)
    params_list = list()
    this_line_length = offset
    line_sep = ',\n' + (1 + offset / 2) * ' '
    for i, (k, v) in enumerate(params.iteritems()):
        if type(v) is float:
            
            this_repr  = '%s=%s' % (k, str(v))
        else:
           
            this_repr  = '%s=%s' % (k, printer(v))
        if len(this_repr) > 500:
            this_repr = this_repr[:300] + '...' + this_repr[-100:]
        if i > 0:
            if (this_line_length + len(this_repr) >= 75
                                        or '\n' in this_repr):
                params_list.append(line_sep)
                this_line_length = len(line_sep)
            else:
                params_list.append(', ')
                this_line_length += 2
        params_list.append(this_repr)
        this_line_length += len(this_repr)

    np.set_printoptions(**options)
    lines = ''.join(params_list)
    
    lines = '\n'.join(l.rstrip(' ') for l in lines.split('\n'))
    return lines


################################################################################
class BaseEstimator(object):
    

    @classmethod
    def _get_param_names(cls):
       
        try:
            args, varargs, kw, default = inspect.getargspec(cls.__init__)
            assert varargs is None, (
                'scikit learn estimators should always specify their '
                'parameters in the signature of their init (no varargs).'
                )

            args.pop(0)
        except TypeError:
           
            args = []
        return args

    def _get_params(self, deep=True):
       
        out = dict()
        for key in self._get_param_names():
            value = getattr (self, key)
            if deep and hasattr (value, '_get_params'):
                deep_items = value._get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def _set_params(self, **params):
       
        if not params:
        
            return
        valid_params = self._get_params(deep=True)
        for key, value in params.iteritems():
            split = key.split('__', 1)
            if len(split) > 1:
               
                name, sub_name = split
                assert name in valid_params, ('Invalid parameter %s '
                                              'for estimator %s' %
                                             (name, self))
                sub_object = valid_params[name]
                assert hasattr(sub_object, '_get_params'), (
                    'Parameter %s of %s is not an estimator, cannot set '
                    'sub parameter %s' %
                        (sub_name, self.__class__.__name__, sub_name)
                    )
                sub_object._set_params(**{sub_name:value})
            else:
                
                assert key in valid_params, ('Invalid parameter %s '
                                              'for estimator %s' %
                                             (key, self.__class__.__name__))
                setattr(self, key, value)
        return self

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (
                class_name,
                _pprint(self._get_params(deep=False),
                        offset=len(class_name),
                ),
            )

    def __str__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (
                class_name,
                _pprint(self._get_params(deep=True),
                        offset=len(class_name),
                        printer=str,
                ),
            )


################################################################################
class ClassifierMixin(object):
   

    def score(self, X, y):
        
        return np.mean(self.predict(X) == y)


################################################################################
class RegressorMixin(object):
   
    def score(self, X, y):
       
        return r2_score(y, self.predict(X))


################################################################################
class TransformerMixin(object):
   
    def fit_transform(self, X, y=None, **fit_params):
       
        if y is None:
            
            return self.fit(X, **fit_params).transform(X)
        else:
           
            return self.fit(X, y, **fit_params).transform(X)

################################################################################

def _get_sub_estimator(estimator):
    
    if hasattr(estimator, 'estimator'):
        
        return _get_sub_estimator(estimator.estimator)
    if hasattr(estimator, 'steps'):
        
        return _get_sub_estimator(estimator.steps[-1][1])
    return estimator


def is_classifier(estimator):
    
    estimator = _get_sub_estimator(estimator)
    return isinstance(estimator, ClassifierMixin)

