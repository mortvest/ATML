import numpy as np

# Based on implementation by Yevgeny Seldin
# function y = ysidkl(x, z)

# Calculates the maximal bias y of a Bernoulli variable such that its
# KL-divergence from a Bernoulli variable with bias x is bounded by z.

# y = argmax_y Dkl(x||y) < z

def ysidkl(x, z, eps=0.0001):
    if ((x < 0) or (x > 1) or (z < 0)):
        raise ValueError('wrong argument')
    elif (z == 0):
        y = x
    else:
        y = (1 + x) / 2
        step = (1 - x) / 4
        if (x > 0):
            p0 = x
        else:
            p0 = 1

        while (step > eps):
            if ((x * np.log(p0 / y) + (1 - x) * np.log((1 - x) / (1 - y))) < z):
                y += step
            else:
                y -= step
            step /= 2
    return y

if __name__ == "__main__":
    print(ysidkl(0.1,0.2))
