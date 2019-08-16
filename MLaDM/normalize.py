import numpy as np

def zero_one(x, axis):
    if axis == 0:
        max = np.max(x, axis=0)
        min = np.min(x, axis=0)
        return (x-min) / (max - min)
        
    elif axis == 1:
        max = np.max(x, axis=1)
        min = np.min(x, axis=1)
        return (x-min) / (max - min)
    else:
        raise ValueError("axis is wrong")
        
def zscore(x, axis):
    if axis == 0：
        _mean = np.mean(x, axis=0)
        _var = np.var(x, axis=0)
        return (x-_mean)/_var
        
    elif axis==1:
        _mean = np.mean(x, axis=1)
        _var = np.var(x, axis=1)
        return (x-_mean)/_var
    else:
        raise ValueError("axis is wrong")