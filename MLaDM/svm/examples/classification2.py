import numpy as N
import pylab as P
import matplotlib as MPL

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )


from .. import svm
import utils

from ..datasets import german
data = german.load()

features = N.vstack([data['data']['feat' + str(i)].astype(N.float) for i in range(1, 25)]).T
label = data['label']

t, s = utils.scale(features)

training = svm.ClassificationDataSet(label, features)

def train_svm(cost, gamma, fold = 5):
    """Train a SVM for given cost and gamma."""
    kernel = svm.kernel.RBF(gamma = gamma)
    model = svm.CClassificationModel(kernel, cost = cost)
    cv = model.cross_validate(training, fold)
    return cv

c_range = N.exp(N.log(2.) * N.arange(-5, 15))
g_range = N.exp(N.log(2.) * N.arange(-15, 3))

# Train the svm on a log distributed grid
gr = N.meshgrid(c_range, g_range)
c = gr[0].flatten()
g = gr[1].flatten()
cf = N.hstack((c, g))
cv = N.empty(c.size)
for i in range(cv.size):
    print "=============== iteration %d / %d ============" % (i, cv.size)
    cv[i] = train_svm(c[i], g[i])

v = P.contourf(N.log2(gr[0]), N.log2(gr[1]), cv.reshape(g_range.size, c_range.size), 10)
v = P.contour(N.log2(gr[0]), N.log2(gr[1]), cv.reshape(g_range.size, c_range.size), 10)
P.clabel(v, inline = 1, fontsize = 10)
P.show()
