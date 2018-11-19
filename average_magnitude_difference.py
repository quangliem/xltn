import numpy as np


def compute_dk(y_or, k):
    if(k>=len(y_or)):
        return 0
    dk=0
    n=0#diem bat dau cua so
    N = len(y_or)
    for n in range(0, N-k):
        dk+= np.abs(y_or[n]-y_or[n+k])
    return dk/(N-k)
def compute_dk_list(y_or, start=1, stop=301):
    return [compute_dk(y_or, k) for k in range(start, stop)]

if __name__ == '__main__':
    pass