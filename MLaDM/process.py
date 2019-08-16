import numpy as np
import data
from naive_bayes import GNB
from lda import LDA
from qda import QDA
from logistic import LogisticRegression 
from pca.pca import PCA





def train_acc(data_path, algorithm_name):
    print(data_path)
    x, y, test_x, test_y = data.run(data_path)
    clf = None
    if algorithm_name == "gnb":
        clf = GNB()
        print "gnb instance."
    elif algorithm_name == "lda" :
        clf = LDA()
        print "lda instance."
    elif algorithm_name == "QDA" :
        clf = QDA()
        print "qda instance."
    elif algorithm_name == "log":
	clf = LogisticRegression()
	print "logistic instance"
    elif algorithm_name == "pca":
        clf = PCA()
        print "PCA instance" 
    else:
        print "NO Implement"
    
    if algorithm_name == "pca":
        clf.fit(x.T, n_components=10)
        print "dimensions original:"
        print x.shape
        print "dimensions PCA processed:"
        print clf.components_.shape
        print "variance PCA processed:"
        print clf.explained_variance_
        return 5
    num = 0
    clf.fit(x, y)
    train_result = clf.predict(x)
    for i in range(len(train_result)):
        if train_result[i] == y[i]:
            num += 1

    return float(num) / len(y)

def test_acc(data_path, algorithm_name):
    print(data_path)
    x, y, test_x, test_y = data.run(data_path)
    clf = None
    if algorithm_name == "gnb":
        clf = GNB()
        print "gnb instance."
    elif algorithm_name == "lda" :
        clf = LDA()
        print "lda instance."
    elif algorithm_name == "QDA" :
        clf = QDA()
        print "qda instance."
    else:
        print "NO Implement"

    num = 0
    clf.fit(x, y)
    train_result = clf.predict(test_x)
    for i in range(len(train_result)):
        if train_result[i] == test_y[i]:
            num += 1
    
    return float(num) / len(test_y)

