import numpy as np
import matplotlib.pyplot as plt

def kl_up_inv(x, z, eps=0.00000001):
    """
    Based on implementation by Yevgeny Seldin
    function y = kl_inv_pos(x, z)
    y = argmax_y Dkl(x||y) < z
    """
    if ((x < 0) or (x > 1) or (z < 0)):
        raise ValueError('wrong argument')
    if (z == 0):
        y = x
    else:
        y = (1 + x) / 2
        step = (1 - x) / 4
        if (x > 0):
            p0 = x
        else:
            p0 = 1
        while (step > eps):
            if (x * np.log(p0 / y) + (1 - x) * np.log((1 - x) / (1 - y))) < z:
                y += step
            else:
                y -= step
            step /= 2
    return y

def kl_low_inv(x, z, eps=0.0000001):
    if ((x < 0) or (x > 1) or (z < 0)):
        raise ValueError('wrong argument')
    y = x
    step = min(x ,1 - x) + eps
    if (x > 0):
        p0 = x
    else:
        p0 = 1
    while step > eps:
        if ((x * np.log(x / y) + (1 - x) * np.log((1 - x) / (1 - y))) < z):
            y -= step
        else:
            y += step
            step /= 2
    return y

def hoeffding(p_hat, n, delta):
    return p_hat + np.sqrt(np.log(1 / delta) / (2 * n))

def hoeffding_low(p_hat, n, delta):
    return p_hat - np.sqrt(np.log(1 / delta)/(2 * n))

def kl(p_val, n, delta):
    z = np.log((n + 1) / delta) / n
    return kl_up_inv(p_val, z)

def kl_low(p_val, n, delta):
    z = np.log((n + 1) / delta) / n
    return kl_low_inv(p_val, z)

def pinsker(p_hat, n, delta):
    return p_hat + np.sqrt(np.log((n + 1) / delta) / (2 * n))

def refined_pinsker(p_hat, n, delta):
    return p_hat + np.sqrt(2 * p_hat * np.log((n + 1) / delta) / n) + 2 * np.log((n + 1) / delta) / n

def main():
    # a
    delta = 0.01
    n = 1000
    p_hat = np.linspace(0,1, num=n)
    hb = hoeffding(p_hat, n, delta)
    plt.plot(p_hat, hb, label="Hoeffding's upper bound")

    # b
    kl_plus = np.array([kl(x, n, delta) for x in p_hat])
    plt.plot(p_hat, kl_plus, label="kl upper bound")

    # c
    kl_pinsker = pinsker(p_hat, n, delta)
    plt.plot(p_hat, kl_pinsker, label="Pinsker's relaxation")

    # d
    ref_pinsker = refined_pinsker(p_hat, n, delta)
    plt.plot(p_hat, ref_pinsker, label="Refined Pinsker's relaxation")

    plt.legend()
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel(r"$\hat{p}_n$")
    plt.ylabel(r"$p$")
    plt.savefig("plt21.png")

    plt.xlim([0,0.1])
    plt.ylim([0,0.20])
    plt.savefig("plt22.png")

    plt.clf()
    kl_minus = np.array([kl_low(x, n, delta) for x in p_hat])
    hb_lower = hoeffding_low(p_hat, n, delta)
    plt.plot(p_hat, hb_lower, label="Hoeffding's lower bound")
    plt.plot(p_hat, kl_minus, label="kl lower bound")

    plt.legend()
    plt.xlim([0,1])
    plt.xlabel(r"$\hat{p}_n$")
    plt.ylabel(r"$p$")
    plt.savefig("plt23.png")

if __name__ == "__main__":
    main()
