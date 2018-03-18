import numpy as np 

a=np.arange(10,110,10)

def moving_average(a, n=4) :
    ret = np.cumsum(a, dtype=int)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


moving_average(a)
#array([25, 35, 45, 55, 65, 75, 85])