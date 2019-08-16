from ctypes import POINTER, c_int, c_double
from itertools import repeat, chain
import numpy as N

from model import Model
import libsvm

izip = zip

__all__ = [
    'CClassificationModel',
    'NuClassificationModel',
    'ClassificationResults'
    ]

class ClassificationResults:
    def __init__(self, model, traindataset, kernel, PredictorType):
        modelc = model.contents
        if modelc.param.svm_type not in [libsvm.C_SVC, libsvm.NU_SVC]:
            raise TypeError('%s is for classification problems' % \
                str(self.__class__))
        self.nr_class = modelc.nr_class
        self.labels = modelc.labels[:self.nr_class]
        nrho = self.nr_class * (self.nr_class - 1) / 2
        self.rho = modelc.rho[:nrho]
        self.nSV = modelc.nSV[:self.nr_class]
        sv_coef = N.empty((self.nr_class - 1, modelc.l), dtype=N.float64)
        for i, c in enumerate(modelc.sv_coef[:self.nr_class - 1]):
            sv_coef[i,:] = c[:modelc.l]
        self.sv_coef = sv_coef
        self.predictor = PredictorType(model, traindataset, kernel)

    def predict(self, dataset):
        
        if self.predictor.is_compact and dataset.is_array_data():
            return [int(x) for x in
                    self.predictor.predict(dataset.data)]
        else:
            return [int(self.predictor.predict(x)) for x in dataset]

    def predict_values(self, dataset):
        
        n = self.nr_class * (self.nr_class - 1) / 2
        def p(vv):
            vv = N.atleast_1d(vv)
            d = {}
            labels = self.labels
            for v, (li, lj) in \
                    izip(vv, chain(*[izip(repeat(x), labels[i+1:])
                                     for i, x in enumerate(labels[:-1])])):
                d[li, lj] = v
                d[lj, li] = -v
            return d
        if self.predictor.is_compact and dataset.is_array_data():
            vs = self.predictor.predict_values(dataset.data, n)
        else:
            vs = [self.predictor.predict_values(x, n) for x in dataset]
        return [p(v) for v in vs]

    def predict_probability(self, dataset):
        
        def p(x):
            n = self.nr_class
            label, prob_estimates = \
                self.predictor.predict_probability(x, self.nr_class)
            return int(label), prob_estimates
        return [p(x) for x in dataset]

    def compact(self):
        self.predictor.compact()

class ClassificationModel(Model):
    

    ResultsType = ClassificationResults

    def __init__(self, kernel, weights, **kwargs):
        Model.__init__(self, kernel, **kwargs)
        if weights is not None:
            self.weight_labels = N.empty((len(weights),), dtype=N.intp)
            self.weights = N.empty((len(weights),), dtype=N.float64)
            weights = weights[:]
            weights.sort()
            for i, (label, weight) in enumerate(weights):
                self.weight_labels[i] = label
                self.weights[i] = weight
            self.param.nr_weight = len(weights)
            self.param.weight_label = \
                self.weight_labels.ctypes.data_as(POINTER(c_int))
            self.param.weight = \
                self.weights.ctypes.data_as(POINTER(c_double))

    def cross_validate(self, dataset, nr_fold):
        
        problem = dataset._create_svm_problem()
        target = N.empty((len(dataset.data),), dtype=N.float64)
        tp = target.ctypes.data_as(POINTER(c_double))
        libsvm.svm_cross_validation(problem, self.param, nr_fold, tp)
        total_correct = 0.
        for x, t in zip(dataset.data, target):
            if x[0] == int(t):
                total_correct += 1
        # XXX also return results from folds in a list
        return 100.0 * total_correct / len(dataset.data)

class CClassificationModel(ClassificationModel):
    

    def __init__(self, kernel,
                 cost=1.0, weights=None, probability=False, **kwargs):
        
        ClassificationModel.__init__(self, kernel, weights, **kwargs)
        self.cost = cost
        self.param.svm_type = libsvm.C_SVC
        self.param.C = cost
        self.param.probability = probability

class NuClassificationModel(ClassificationModel):
    

    def __init__(self, kernel,
                 nu=0.5, weights=None, probability=False, **kwargs):
        
        ClassificationModel.__init__(self, kernel, weights, **kwargs)
        self.nu = nu
        self.param.svm_type = libsvm.NU_SVC
        self.param.nu = nu
        self.param.probability = probability
