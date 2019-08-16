
import numpy as N
import pylab as P
import matplotlib as MPL

import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from ..machine import svm
import utils

data = utils.iris.load()

def get_data(xattr, yattr, ln):
    """Given x and y attributes, returns label, samples, label, sample, where
    the first couple (label, samples) is for training, the other one for
    testing.

    For each class, the first nl samples are used for training, the other for testing.
    """
    lxdata = {}
    lydata = {}
    txdata = {}
    tydata = {}
    llabel = {}
    tlabel = {}

    data = utils.iris.load()
    cnames = data.keys()
    lnames = {}
    for i in range(len(cnames)):
        lnames[cnames[i]] = i

    for i in cnames:
        lxdata[i] = data[i][xattr][:ln]
        txdata[i] = data[i][xattr][ln:]
        lydata[i] = data[i][yattr][:ln]
        tydata[i] = data[i][yattr][ln:]

    lxdata = N.concatenate([lxdata[i] for i in cnames])
    lydata = N.concatenate([lydata[i] for i in cnames])
    txdata = N.concatenate([txdata[i] for i in cnames])
    tydata = N.concatenate([tydata[i] for i in cnames])

    llabel = N.concatenate([lnames[i] * N.ones(ln, N.int) for i in cnames])
    tlabel = N.concatenate([lnames[i] * N.ones(ln, N.int) for i in cnames])

    return llabel, N.vstack((lxdata, lydata)).T, tlabel, \
           N.vstack((txdata, tydata)).T, cnames, lnames

#--------------------
ln = 25
llabel, ldata, tlabel, tdata, cnames, lnames = get_data('sepal width', 'petal width', ln)

training = svm.LibSvmClassificationDataSet(llabel, ldata)
testing = svm.LibSvmTestDataSet(tdata)

def train_svm(cost, gamma, fold = 5):
    """Train a SVM for given cost and gamma."""
    kernel = svm.RBFKernel(gamma = gamma)
    model = svm.LibSvmCClassificationModel(kernel, cost = cost)
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
    cv[i] = train_svm(c[i], g[i])

v = P.contour(N.log2(gr[0]), N.log2(gr[1]), cv.reshape(g_range.size, c_range.size), 12)
P.clabel(v, inline = 1, fontsize = 10)
P.show()
P.legend()
