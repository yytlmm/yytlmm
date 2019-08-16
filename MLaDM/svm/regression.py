from ctypes import POINTER, c_double
import numpy as N

from model import Model
import libsvm

__all__ = [
    'EpsilonRegressionModel',
    'NuRegressionModel',
    'RegressionResults'
    ]

# XXX document why get_svr_probability could be useful

class RegressionResults:
    def __init__(self, model, traindataset, kernel, PredictorType):
        modelc = model.contents
        if modelc.param.svm_type not in [libsvm.EPSILON_SVR, libsvm.NU_SVR]:
            raise TypeError, '%s is for regression problems' % \
                str(self.__class__)
        self.rho = modelc.rho[0]
        self.sv_coef = modelc.sv_coef[0][:modelc.l]
        if modelc.probA:
            self.sigma = modelc.probA[0]
        self.predictor = PredictorType(model, traindataset, kernel)

    def predict(self, dataset):
       
        z = [self.predictor.predict(x) for x in dataset]
        return N.asarray(z).squeeze()

    def get_svr_probability(self):
        
        return self.sigma

    def compact(self):
        self.predictor.compact()

class RegressionModel(Model):
    ResultsType = RegressionResults

    def __init__(self, kernel, **kwargs):
        Model.__init__(self, kernel, **kwargs)

    def cross_validate(self, dataset, nr_fold):
        
        problem = dataset._create_svm_problem()
        target = N.empty((len(dataset.data),), dtype=N.float64)
        tp = target.ctypes.data_as(POINTER(c_double))
        libsvm.svm_cross_validation(problem, self.param, nr_fold, tp)

        total_error = sumv = sumy = sumvv = sumyy = sumvy = 0.
        for i in range(len(dataset.data)):
            v = target[i]
            y = dataset.data[i][0]
            sumv = sumv + v
            sumy = sumy + y
            sumvv = sumvv + v * v
            sumyy = sumyy + y * y
            sumvy = sumvy + v * y
            total_error = total_error + (v - y) * (v - y)

        # mean squared error
        mse = total_error / len(dataset.data)
        # squared correlation coefficient
        l = len(dataset.data)
        scc = ((l * sumvy - sumv * sumy) * (l * sumvy - sumv * sumy)) / \
            ((l * sumvv - sumv*sumv) * (l * sumyy - sumy * sumy))

        return mse, scc

class EpsilonRegressionModel(RegressionModel):
   
    def __init__(self, kernel, epsilon=0.1, cost=1.0,
                 probability=False, **kwargs):
        RegressionModel.__init__(self, kernel, **kwargs)
        self.epsilon = epsilon
        self.cost = cost
        self.param.svm_type = libsvm.EPSILON_SVR
        self.param.p = epsilon
        self.param.C = cost
        self.param.probability = probability

class NuRegressionModel(RegressionModel):
    
    def __init__(self, kernel,
                 nu=0.5, cost=1.0, probability=False, **kwargs):
        RegressionModel.__init__(self, kernel, **kwargs)
        self.nu = nu
        self.cost = cost
        self.param.svm_type = libsvm.NU_SVR
        self.param.nu = nu
        self.param.C = cost
        self.param.probability = probability
