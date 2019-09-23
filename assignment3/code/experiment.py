"""
Some this code is borrowed from my implementation of the Task 6.4 for the
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
    """Weak SVM implementation with gamma, chosen randomly from the grid"""
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
    """
    Discrete aggregation of multiple classifiers
    """
    signs = np.sign(votes.T @ rho).astype(int)
    signs[signs == 0] = 1
    return signs


def collect_weak_data(m, train_x, train_y, test_x, test_y, param_grid):
    """
    Initialize m weak SVM classifiers, train them and collect losses,
    votes and runtimes of each
    """
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
    """
    Uniform distribution with for m values
    """
    return np.repeat(1.0/m, m)


def KL_div(p,q):
    """
    KL divergence of p and q
    """
    return np.sum(p * np.log(p/q))


def kl_up_inv(x, z, eps=0.00000001):
    """
    Inversion of kl using binary search. Based on implementation by Yevgeny Seldin
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


def find_rho(L_hat, n, r, m, delta):
    """
    Implementation of the alternating minimization of PAC-Bayes-Gamma bound
    """
    def alt_minimize():
        """
        update step
        """
        nonlocal rho
        nonlocal lmbda
        lnr = -lmbda * (n - r)
        rho = pi * np.exp(lnr * L_hat) / (np.sum(pi * np.exp(lnr * L_hat)))
        exp_loss = np.sum(rho * L_hat)
        klrp = KL_div(rho, pi)
        lmbda = 2 / (np.sqrt(
            (2 * (n - r) * exp_loss) / (klrp + np.log((2 * np.sqrt(n - r) / delta)) + 1)) + 1)

    # initiate timer
    start = time.perf_counter()
    # initiate rho and pi with uniform distributions
    pi = uniform_dist(m)
    rho = uniform_dist(m)
    # initiate variables
    lmbda = 1
    diff = rho
    old = 1
    count = 0
    # threshold
    eps = 0.00001
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


def majority_vote(m, n, train_x, train_y, test_x, test_y, grid_params, r, delta):
    test_y_m = np.copy(test_y)
    test_y_m[test_y_m == 0] = -1

    losses, vote_matrix, times = collect_weak_data(m, train_x, train_y, test_x, test_y, grid_params)
    rho, time_rho = find_rho(losses, n, r, m, delta)
    agg_alg = aggregate(vote_matrix, rho)
    acc = accuracy_score(test_y_m, agg_alg)
    time_tot = np.sum(times) + time_rho
    if debug:
        print("score algorithm for {}: {}".format(m, acc))
    return acc, time_tot, rho


def pac_bayes_kl(L_hat, rho, n, m, r, delta):
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


def run_once(n, ms, n_ticks, data, r, delta):
    # shuffle dataset
    np.random.shuffle(data)

    # split data
    data_x = data[:,:-1]
    data_y = data[:,-1].astype(int)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, train_size=n)
    assert(train_x.shape[0] + test_x.shape[0] == data.shape[0]), "Split is incorrect"

    # calculate parameter grid
    grid_params = jaakkola_grid(train_x, train_y)

    # baseline CV SVM
    bl_svm = BaselineSVM()
    bl_train_acc = bl_svm.train(train_x, train_y, grid_params)
    bl_test_preds, bl_test_acc = bl_svm.test(test_x, test_y)
    bl_time = bl_svm.time

    # PAC Bayes Aggregation SVMs
    accuracies = []
    times = []
    rhos = []
    for m in ms:
        a, t, rho = majority_vote(m, n, train_x, train_y, test_x, test_y, grid_params, r, delta)
        accuracies.append(a)
        times.append(t)
        rhos.append(rho)
    losses = 1 - np.array(accuracies)

    # apply bound
    bound = [pac_bayes_kl(L_hat, rho, n, m, r, delta) for L_hat, rho, m in zip(losses, rhos, ms)]
    return losses, np.array(times), np.array(bound), (1- bl_test_acc), bl_time


def main():
    # np.random.seed(420)
    # load data
    print("Loading data")
    data_folder = "./data/"
    data = np.loadtxt(data_folder + "ionosphere.data", delimiter=",")

    # define constants
    n_ticks = 20
    n_tries = 25
    n = 200
    r = data.shape[1]
    delta = 0.05
    ms = np.logspace(0.3, np.log10(n), num=n_ticks, endpoint=True).astype(int)
    if debug: print(ms)

    # collecting results
    losses_s = []
    times_s = []
    bound_s = []
    bl_test_s = []
    bl_time_s = []
    count = 1
    for _ in range(n_tries):
        print("Running experiment #{}".format(count))
        one, two, three, four, five = run_once(n, ms, n_ticks, data, r, delta)
        losses_s.append(one)
        times_s.append(two)
        bound_s.append(three)
        bl_test_s.append(four)
        bl_time_s.append(five)
        count += 1

    # calculating metrics
    print("Tests ran. Calculating metrics and plotting")
    losses = np.mean(losses_s, axis=0)
    times = np.mean(times_s, axis=0)
    bound = np.mean(bound_s, axis=0)
    bl_test_acc = np.mean(bl_test_s, axis=0)
    bl_time = np.mean(bl_time_s, axis=0)

    loss_std = np.std(losses_s)
    times_std = np.std(times_s)
    bl_test_acc_std = np.std(bl_test_s)
    bl_time_std = np.std(bl_time_s)

    print("Standard deviations:")
    print("PAC Bayes loss:", loss_std)
    print("CV loss:", bl_test_acc_std)
    print("PAC Bayes time:", times_std)
    print("CV time:", bl_time_std)

    # plotting
    fig, ax1 = plt.subplots()

    ax1.set_xlabel('m')
    ax1.set_ylabel('Test loss')
    ax1.plot(ms, losses, color="black", label="Our method")
    ax1.plot(ms, np.repeat(bl_test_acc, n_ticks), color="red", label="CV SVM")
    ax1.plot(ms, bound, color="blue", label="Bound")
    ax1.set_xscale('log')

    ax2 = ax1.twinx()

    ax2.set_ylabel("Runtime(s)")
    ax2.plot(ms, np.array(times), color="black", linestyle="--", label=r"$t_m$")
    ax2.plot(ms, np.repeat(bl_time, n_ticks), color="red", linestyle="--", label=r"$t_{CV}$")
    ax2.set_xscale('log')

    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.92))
    plt.savefig("plt1.png")

if __name__ == "__main__":
    debug = False
    main()
