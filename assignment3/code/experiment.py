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
        clf = GridSearchCV(svc, grid_params, cv=5)
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


def find_gamma(sigma):
    return 1.0/(2.0 * sigma**2.0)

def exp_b_lst(b, lst):
    return b ** lst

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

def find_rho(L_hat, n, r, m):
    def alt_minimize():
        nonlocal rho
        nonlocal lmbda
        lnr = -lmbda * (n - r)
        rho = pi * np.exp(lnr * L_hat) / (np.sum(pi * np.exp(lnr * L_hat)))
        exp_loss = np.sum(rho * L_hat)
        delta = 0.05
        klrp = np.sum(rho * np.log(rho/pi))
        lmbda = 2 / (np.sqrt((2 * n * exp_loss)
                             / (klrp + np.log((n + 1) / delta)) + 1) + 1)

    start = time.perf_counter()
    pi = uniform_dist(m)
    rho = uniform_dist(m)
    lmbda = 0.5
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

def weak_method(m, n, train_x, train_y, test_x, test_y, grid_params):
    r = 35
    test_y_m = np.copy(test_y)
    test_y_m[test_y_m == 0] = -1

    losses, vote_matrix, times = collect_weak_data(m, train_x, train_y, test_x, test_y, grid_params)
    rho, time_rho = find_rho(losses, n, r, m)
    agg_alg = aggregate(vote_matrix, rho)
    acc = accuracy_score(test_y_m, agg_alg)
    time_tot = np.sum(times) + time_rho
    if debug:
        print("score algorithm for {}: {}".format(m, acc))
    return acc, time_tot


def results():
    # np.random.seed(420)

    data_folder = "./data/"
    data = np.loadtxt(data_folder + "ionosphere.data",
                      delimiter = ",")
    if debug: print(np.shape(data))
    np.random.shuffle(data)
    n = 200
    data_x = data[:,:-1]
    data_y = data[:,-1].astype(int)
    if debug:
        print(data_x[0])
        print(data_y[0])

    train_x, test_x, train_y, test_y = train_test_split(
        data_x, data_y, train_size=n)

    assert(train_x.shape[0] + test_x.shape[0] == data.shape[0]), "Split is incorrect"

    if debug: print(train_y, test_y)
    train_sub_0s = train_x[train_y == 0]
    train_sub_1s = train_x[train_y == 1]

    # applying Jaakkola's heuristic
    sigma_jak = find_jaakkola(train_sub_0s, train_sub_1s)
    gamma_jak = find_gamma(sigma_jak)
    # grid parameters
    grid_b = 10.0
    Cs = np.array([grid_b**x for x in range(-3,4)])
    if debug: print(Cs)
    grid_params = [
        {'C': Cs,
         'gamma': gamma_jak * exp_b_lst(grid_b, np.arange(-4,5))},
    ]

    bl_svm = BaselineSVM()
    bl_train_acc = bl_svm.train(test_x, test_y, grid_params)
    bl_test_preds, bl_test_acc = bl_svm.test(test_x, test_y)
    bl_time = bl_svm.time

    print(bl_test_acc)

    accuracies = []
    times = []
    n_ticks = 20
    ms = np.linspace(1,n, num=n_ticks, endpoint=True).astype(int)
    for m in ms:
        a, t = weak_method(m, n, train_x, train_y, test_x, test_y, grid_params)
        accuracies.append(a)
        times.append(t)

    # plt.plot(ms, (1 - np.array(accuracies)))
    # plt.plot(ms, (1 - np.repeat(bl_test_acc, n_ticks)))
    # plt.show()

    fig, ax1 = plt.subplots()

    ax1.set_xlabel('m')
    ax1.set_ylabel('Test loss')
    ax1.plot(ms, (1 - np.array(accuracies)), color="black")
    ax1.plot(ms, (1 - np.repeat(bl_test_acc, n_ticks)), color="red")
    ax1.tick_params(axis='y')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel("Runtime(s)")  # we already handled the x-label with ax1
    ax2.plot(ms, np.array(times), color="black", linestyle="--")
    ax2.plot(ms, np.repeat(bl_time, n_ticks), color="red", linestyle="--")
    ax2.tick_params(axis="y")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

if __name__ == "__main__":
    debug = False
    results()
