
import numpy as np
import sys
from time import time


class LossFunction:
    """Base class for convex loss functions"""

    def loss(self, p, y):
        """Evaluate the loss function.

        :arg p: The prediction.
        :type p: double
        :arg y: The true value.
        :type y: double
        :returns: double"""
        raise NotImplementedError()

    def dloss(self,  p,  y):
        """Evaluate the derivative of the loss function.

        :arg p: The prediction.
        :type p: double
        :arg y: The true value.
        :type y: double
        :returns: double"""
        raise NotImplementedError()


class Regression(LossFunction):
    """Base class for loss functions for regression"""

    def loss(self, p, y):
        raise NotImplementedError()

    def dloss(self, p, y):
        raise NotImplementedError()


class Classification(LossFunction):
    """Base class for loss functions for classification"""

    def loss(self,  p, y):
        raise NotImplementedError()

    def dloss(self, p, y):
        raise NotImplementedError()


class ModifiedHuber(Classification):
    """Modified Huber loss for binary classification with y in {-1, 1}

    This is equivalent to quadratically smoothed SVM with gamma = 2.

    See T. Zhang 'Solving Large Scale Linear Prediction Problems Using
    Stochastic Gradient Descent', ICML'04.
    """
    def loss(self, p, y):
        z = p * y
        if z >= 1.0:
            return 0.0
        elif z >= -1.0:
            return (1.0 - z) * (1.0 - z)
        else:
            return -4.0 * z

    def dloss(self, p, y):
        z = p * y
        if z >= 1.0:
            return 0.0
        elif z >= -1.0:
            return 2.0 * (1.0 - z) * y
        else:
            return 4.0 * y

    def __reduce__(self):
        return ModifiedHuber, ()



class Hinge(Classification):
    """SVM loss for binary classification tasks with y in {-1,1}"""
    def loss(self, p, y):
        z = p * y
        if z < 1.0:
            return (1 - z)
        return 0.0

    def dloss(self, p, y):
        z = p * y
        if z < 1.0:
            return y
        return 0.0

    def __reduce__(self):
        return Hinge, ()

class Log(Classification):
    """Logistic regression loss for binary classification with y in {-1, 1}"""

    def loss(self, p, y):
        z = p * y
        # approximately equal and saves the computation of the log
        if z > 18:
            return np.exp(-z)
        if z < -18:
            return -z * y
        return np.log(1.0 + np.exp(-z))

    def dloss(self, p, y):
        z = p * y
        # approximately equal and saves the computation of the log
        if z > 18.0:
            return np.exp(-z) * y
        if z < -18.0:
            return y
        return y / (np.exp(z) + 1.0)

    def __reduce__(self):
        return Log, ()


class SquaredLoss(Regression):
    """Squared loss traditional used in linear regression."""
    def loss(self, p, y):
        return 0.5 * (p - y) * (p - y)

    def dloss(self, p, y):
        return y - p

    def __reduce__(self):
        return SquaredLoss, ()


class Huber(Regression):
    """Huber regression loss

    Variant of the SquaredLoss that is robust to outliers (quadratic near zero,
    linear in for large errors).

    References
    ----------

    http://en.wikipedia.org/wiki/Huber_Loss_Function
    """

    def __init__(self,c):
        self.c = c

    def loss(self, p, y):
        r = p - y
        abs_r = abs(r)
        if abs_r <= self.c:
            return 0.5 * r * r
        else:
            return self.c * abs_r - (0.5 * self.c * self.c)

    def dloss(self, p, y):
        r = y - p
        abs_r = abs(r)
        if abs_r <= self.c:
            return r
        elif r > 0.0:
            return self.c
        else:
            return -self.c

    def __reduce__(self):
        return Huber,(self.c,)

