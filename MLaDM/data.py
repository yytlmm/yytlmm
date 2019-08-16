import numpy as np 
import pandas as pd


def shuffle_train_data(data):
    shuffled_indices = np.random.permutation(len(data))
    return data.iloc[shuffled_indices]

def run(data_path):
    train = pd.read_csv(data_path + "/mnistTrain_scale.txt", header= None)
    test = pd.read_csv(data_path + "/mnistTest_scale.txt", header= None)
    
    a = train[train[784]==7]
    b = train[train[784]==1]
    train_result = pd.concat([a,b])
    
    c = test[test[784] == 1]
    d = test[test[784] == 7]
    test_result = pd.concat([c,d])
    
    # test.
    train = np.array(shuffle_train_data(train_result))
    test = np.array(shuffle_train_data(test_result))
    
    # X_train : (13007, 784)
    # Y_train : (13007, )
    X_train = train[:,:-1]
    Y_train = train[:,-1]
    
    # X_test : (2163, 784)
    # Y_test : (2163,)
    X_test = test[:,:-1]
    Y_test = test[:,-1]

    for i in range(len(Y_train)):
        if Y_train[i] == 7:
            Y_train[i] = 0

    for i in range(len(Y_test)):
        if Y_test[i] == 7:
            Y_test[i] = 0

    return X_train, Y_train, X_test, Y_test
