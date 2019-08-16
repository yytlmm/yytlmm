import numpy as np

def euclidean_distances(X, Y, Y_norm_squared=None, squared=False):
    
    if X is Y:
        X = Y = np.asanyarray(X)
    else:
        X = np.asanyarray(X)
        Y = np.asanyarray(Y)

    if X.shape[1] != Y.shape[1]:
        raise ValueError("Incompatible dimension for X and Y matrices")

    XX = np.sum(X * X, axis=1)[:, np.newaxis]
    if X is Y: 
        YY = XX.T
    elif Y_norm_squared is None:
        YY = Y.copy()
        YY **= 2
        YY = np.sum(YY, axis=1)[np.newaxis, :]
    else:
        YY = np.asanyarray(Y_norm_squared)
        if YY.shape != (Y.shape[0],):
            raise ValueError("Incompatible dimension for Y and Y_norm_squared")
        YY = YY[np.newaxis, :]

    distances = XX + YY 
    distances -= 2 * np.dot(X, Y.T)
    distances = np.maximum(distances, 0)
    if squared:
        return distances
    else:
        return np.sqrt(distances)

euclidian_distances = euclidean_distances 