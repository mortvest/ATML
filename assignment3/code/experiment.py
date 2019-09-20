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
        end = time.perf_counter()
        self.time = end - start
        self.clf = clf
        return self.clf.score(train_x, train_y)

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
        end = time.perf_counter()
        self.time = end - start
        self.clf = clf
        return self.clf.score(new_valid_x, new_valid_y)


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


def collect_votes(m, train_x, train_y, test_x, test_y, param_grid):
    losses = []
    votes = []
    for _ in range(m):
        wsvm = WeakSVM()
        loss = wsvm.train(train_x, train_y, param_grid)
        losses.append(1.0 - loss)
        vote, _ = wsvm.test(test_x, test_y)
        votes.append(vote)
    return np.array(losses), np.array(votes)

def uniform_dist(m):
    return np.repeat(1.0/m, m)

def test():
    M = np.array([
        [1,-1,1,1,1,-1,-1],
        [-1,1,-1,1,1,-1,-1],
        [1,-1,-1,-1,1,-1,-1],
        [-1,-1,1,1,1,-1,-1]
    ])
    rho = np.repeat(1/4, 4)
    agg = aggregate(M, rho)
    print(agg)


def results():
    np.random.seed(420)

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

    print("Baseline train accuracy:", bl_train_acc)
    print("Baseline test accuracy:", bl_test_acc)
    print("Baseline time:", bl_time)
    print("Baseline preds:", bl_test_preds)
    m = 100
    losses, vote_matrix = collect_votes(m, train_x, train_y, test_x, test_y, grid_params)
    print(losses)
    rho = uniform_dist(m)
    agg = aggregate(vote_matrix, rho)
    print(vote_matrix.shape)
    # for i in range(0, 150, 2):
    #     print(vote_matrix[i])
    # print(rho.shape)
    print(agg)

if __name__ == "__main__":
    debug = False
    results()
    # test()
