import numpy as np


def compute_arcf(y_or, k):
    if(k>=len(y_or)):
        return 0
    rs = 0
    N = len(y_or)
    for n in range(0, N-k):
        rs+=y_or[n]*y_or[n+k]
    return rs
def compute_arcf_list(y_or, start=0, stop=301):
    return [compute_arcf(y_or, k) for k in range(start, stop)]

if __name__ == '__main__':
    pass