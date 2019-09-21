"""
A lot of this code is borrowed from my implementation of the Task 6.4 for the
final exam assignment in Machine Learning at DIKU 2018
"""
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score


class SVM():
    """ Abstract class. Implement an instance for each type of SVM """
    def __init__(self):
        self.clf = None
        self.time = None

    def train(self, train_x, train_y, grid_params):
        pass

    def test(self, test_x, test_y):
        preds = self.clf.predict(test_x)
        preds[preds == 0] = -1
        return preds, self.clf.score(test_x, test_y)


class BaselineSVM(SVM):
    """Strong SVM implementation with CV"""
    def train(self, train_x, train_y, grid_params):
        start = time.perf_counter()
        svc = SVC()
        clf = GridSearchCV(svc, grid_params, cv=5, iid=False)
        clf.fit(train_x, train_y)
        validation = clf.score(train_x, train_y)
        end = time.perf_counter()
        self.time = end - start
        self.clf = clf
        return validation


class WeakSVM(SVM):
    """Weak SVM implementation"""
    def train(self, train_x, train_y, grid_params):
        r = train_x.shape[1] + 1
        # split training set into r and n-r
        new_train_x, new_valid_x, new_train_y, new_valid_y = train_test_split(
            train_x, train_y, train_size=r)
        # choose random gamma
        gamma = np.random.choice(grid_params[0]["gamma"])
        start = time.perf_counter()
        clf = SVC(gamma=gamma)
        clf.fit(new_train_x, new_train_y)
        validation = clf.score(new_valid_x, new_valid_y)
        end = time.perf_counter()
        self.time = end - start
        self.clf = clf
        return validation


def find_jaakkola(zeros, ones):
    def find_mins(fst, snd):
        dists = cdist(fst, snd)
        return np.amin(dists, axis=0)
    mins_z = find_mins(zeros, ones)
    mins_o = find_mins(ones, zeros)
    return np.median(np.concatenate((mins_z, mins_o)))


def aggregate(votes, rho):
    signs = np.sign(votes.T @ rho).astype(int)
    signs[signs == 0] = 1
    return signs


def collect_weak_data(m, train_x, train_y, test_x, test_y, param_grid):
    losses = []
    votes = []
    times = []
    for _ in range(m):
        wsvm = WeakSVM()
        loss = wsvm.train(train_x, train_y, param_grid)
        losses.append(1.0 - loss)
        vote, _ = wsvm.test(test_x, test_y)
        votes.append(vote)
        times.append(wsvm.time)
    return np.array(losses), np.array(votes), np.array(times)


def uniform_dist(m):
    return np.repeat(1.0/m, m)

def KL_div(p,q):
    return np.sum(p * np.log(p/q))

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

def find_rho(L_hat, n, r, m):
    def alt_minimize():
        nonlocal rho
        nonlocal lmbda
        lnr = -lmbda * (n - r)
        rho = pi * np.exp(lnr * L_hat) / (np.sum(pi * np.exp(lnr * L_hat)))
        exp_loss = np.sum(rho * L_hat)
        delta = 0.05
        klrp = KL_div(rho, pi)
        lmbda = 2 / (np.sqrt(
            (2 * n * exp_loss)
            # / (klrp + np.log((2*np.sqrt(n) / delta)) + 1)) + 1)
            / (klrp + np.log((n + 1) / delta)) + 1) + 1)

    # initiate timer
    start = time.perf_counter()
    # initiate distributions
    pi = uniform_dist(m)
    rho = uniform_dist(m)
    # initiate variables
    lmbda = 1
    eps = 0.00001
    diff = rho
    old = 1
    count = 0
    while np.sum(np.abs(diff)) > eps:
        count += 1
        old = np.copy(rho)
        old_l = lmbda
        alt_minimize()
        diff = old - rho
        if debug:
            print("delta rho:", np.sum(np.abs(diff)))
            print("delta lambda:", np.abs(old_l - lmbda))

    end = time.perf_counter()
    time_rho = end - start
    if debug: print("# of iterations:", count)
    return rho, time_rho


def majority_vote(m, n, train_x, train_y, test_x, test_y, grid_params, r=35):
    test_y_m = np.copy(test_y)
    test_y_m[test_y_m == 0] = -1

    losses, vote_matrix, times = collect_weak_data(m, train_x, train_y, test_x, test_y, grid_params)
    rho, time_rho = find_rho(losses, n, r, m)
    agg_alg = aggregate(vote_matrix, rho)
    acc = accuracy_score(test_y_m, agg_alg)
    time_tot = np.sum(times) + time_rho
    if debug:
        print("score algorithm for {}: {}".format(m, acc))
    return acc, time_tot, rho


def pac_bayes_kl(L_hat, rho, n, m, delta=0.05, r=35):
    pi = uniform_dist(m)
    z = (KL_div(rho, pi) + np.log((2 * np.sqrt(n-r)) / delta))/(n-r)
    return kl_up_inv(L_hat, z)

def jaakkola_grid(train_x, train_y):
    train_sub_0s = train_x[train_y == 0]
    train_sub_1s = train_x[train_y == 1]

    # applying Jaakkola's heuristic
    sigma_jak = find_jaakkola(train_sub_0s, train_sub_1s)
    gamma_jak = 1.0 / (2.0 * sigma_jak**2.0)
    # grid parameters
    grid_b = 10.0
    grid_params = [
        {'C': np.array([grid_b ** x for x in range(-3,4)]),
         'gamma': np.array([gamma_jak * grid_b ** x for x in np.arange(-4,5)])}
    ]
    return grid_params


def run_once(n, ms, n_ticks, train_x, train_y, test_x, test_y, grid_params):
    accuracies = []
    times = []
    rhos = []
    for m in ms:
        a, t, rho = majority_vote(m, n, train_x, train_y, test_x, test_y, grid_params)
        accuracies.append(a)
        times.append(t)
        rhos.append(rho)

    losses = 1 - np.array(accuracies)
    bound = [pac_bayes_kl(L_hat, rho, n, m) for L_hat, rho, m in zip(losses, rhos, ms)]
    return losses, np.array(times), np.array(bound)


def results():
    # np.random.seed(420)

    data_folder = "./data/"
    data = np.loadtxt(data_folder + "ionosphere.data", delimiter=",")
    if debug: print(np.shape(data))
    np.random.shuffle(data)
    n = 200
    data_x = data[:,:-1]
    data_y = data[:,-1].astype(int)

    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size=n)
    assert(train_x.shape[0] + test_x.shape[0] == data.shape[0]), "Split is incorrect"

    grid_params = jaakkola_grid(train_x, train_y)

    if debug: print(train_y, test_y)
    bl_svm = BaselineSVM()
    bl_train_acc = bl_svm.train(test_x, test_y, grid_params)
    bl_test_preds, bl_test_acc = bl_svm.test(test_x, test_y)
    bl_time = bl_svm.time

    # accuracies = []
    # times = []
    # rhos = []
    # n_ticks = 20
    # ms = np.logspace(0.1, np.log10(n), num=n_ticks, endpoint=True).astype(int)
    # for m in ms:
    #     a, t, rho = majority_vote(m, n, train_x, train_y, test_x, test_y, grid_params)
    #     accuracies.append(a)
    #     times.append(t)
    #     rhos.append(rho)

    # accuracies, times, rhos = run_once(n, train_x, train_y, test_x, test_y, grid_params)
    # losses = 1 - np.array(accuracies)
    # bound = [pac_bayes_kl(L_hat, rho, n, m) for L_hat, rho, m in zip(losses, rhos, ms)]
    n_ticks = 20
    n_tries = 75
    ms = np.logspace(0.1, np.log10(n), num=n_ticks, endpoint=True).astype(int)

    losses_s, times_s, bound_s = run_once(n, ms, n_ticks, train_x, train_y, test_x, test_y, grid_params)
    for _ in range(n_tries-1):
        one, two, three = run_once(n, ms, n_ticks, train_x, train_y, test_x, test_y, grid_params)
        losses_s += one
        times_s += two
        bound_s += three

    losses = losses_s / n_tries
    times = times_s / n_tries
    bound = bound_s /n_tries

    # plotting
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('m')
    ax1.set_ylabel('Test loss')
    ax1.plot(ms, losses, color="black", label="Our method")
    ax1.plot(ms, (1 - np.repeat(bl_test_acc, n_ticks)), color="red", label="CV SVM")
    ax1.plot(ms, bound, color="blue", label="Bound" )
    ax1.set_xscale('log')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel("Runtime(s)")  # we already handled the x-label with ax1
    ax2.plot(ms, np.array(times), color="black", linestyle="--", label=r"$t_m$")
    ax2.plot(ms, np.repeat(bl_time, n_ticks), color="red", linestyle="--", label=r"$t_{CV}$")
    ax2.set_xscale('log')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.92))
    # plt.show()
    plt.savefig("plt1.png")

if __name__ == "__main__":
    debug = False
    results()
