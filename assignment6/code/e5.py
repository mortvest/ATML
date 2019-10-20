import numpy as np
import matplotlib.pyplot as plt


class Plot():
    """abstract class"""
    def __init__(self, T, mu_star, mu, K, n_reps):
        self.T = T
        self.mu_star = mu_star
        self.mu = mu
        self.K = K
        self.n_reps = n_reps

    def label(self):
        pass
    def plot(self):
        acc = np.zeros(self.T)
        xs = np.linspace(1, self.T, self.T, dtype=int)
        for n in range(self.n_reps):
            x = self.calc_regrets(self.T, self.K, self.mu_star, self.mu)
            acc += x
        ys = acc/self.n_reps
        plt.plot(xs, ys, label=self.label())

    def calc_regrets(self, T, K, mu_star, mu):
        """implement for each type of plot"""
        pass
    def print_status(self):
        print("{} for K={}, mu={}".format(self.label(), self.K, self.mu))

class UCB1(Plot):
    """generic UCB1"""
    def calc_regrets(self, T, K, mu_star, mu):
        return self.__UCB1_generic(T, K, mu_star, mu, self.bound)
    def bound(self, t, Ns):
        """implement for each type of UCB1"""
        pass
    def __UCB1_generic(self, T, K, mu_star, mu, bound):
        # initialization
        init_avg = [one_zero_with_prob(mu_star)]
        for _ in range(K-1):
            init_avg.append(one_zero_with_prob(mu))
        avg_rewards = np.array(init_avg).astype(float)
        probs = np.array([mu_star] + [mu] * (K-1))
        Ns = np.ones(K)
        regrets = np.zeros(T)
        # normal runs
        for t in range(1, T+1):
            # apply the bound
            curr_avgs = bound(t, Ns) + avg_rewards
            # find the best hand
            best_ind = np.argsort(curr_avgs)[-1]
            # play the hand
            gain = one_zero_with_prob(probs[best_ind])
            N_old = Ns[best_ind]
            avg_old = avg_rewards[best_ind]
            # update average
            avg_rewards[best_ind] = (avg_old * N_old + gain) / float((N_old + 1))
            # update times played
            Ns[best_ind] += 1
            # calculate the regret
            emp_reg = empirical_regret(Ns, (mu_star - probs))
            regrets[t-1] = emp_reg
        return regrets


class UCB1_notes(UCB1):
    def label(self):
        return "UCB1 from the lecture notes"
    def bound(self, t, N):
        return np.sqrt((3 * np.log(t))/(2 * N))

class UCB1_improved(UCB1):
    def label(self):
        return "UCB1 improved"
    def bound(self, t, N):
        return np.sqrt(np.log(t)/N)

class EXP3(Plot):
    def label(self):
        return "EXP3"
    def __eta_t(self, K, t):
        return np.sqrt(np.log(K)/(t * K))
    def test(self):
        return self.calc_regrets(self.T, self.K, self.mu_star, self.mu)
    def calc_regrets(self, T, K, mu_star, mu):
        # init data
        Ls = np.zeros(K)
        ps = np.zeros(K)
        probs = np.array([mu_star] + [mu] * (K-1))
        # number of time each hand was played
        Ns = np.ones(K)
        regrets = np.zeros(T)
        # iteration
        for t in range(1, T+1):
            # calculate learning rate
            eta = self.__eta_t(K, t)
            # calculate ps:
            ex = - eta * Ls
            # XXX Does this work?
            ps = np.exp(ex)/(np.sum(np.exp(ex)))
            # sample hand based on p
            hand = np.random.choice(ps.shape[0], p=ps)
            # play the hand
            gain = 1-one_zero_with_prob(probs[hand])
            l_wave = gain/ps[hand]
            # update Ls
            Ls[hand] += l_wave
            Ns[hand] += 1
            emp_reg = empirical_regret(Ns, (mu_star - probs))
            regrets[t-1] = emp_reg
        return regrets


def empirical_regret(N_ts, deltas):
    return np.sum(N_ts * deltas)

def one_zero_with_prob(prob):
    return np.random.choice(2, p=[1-prob, prob])

def plot(T, mu_star, K, mu, n_reps):
    if debug: print("Plotting for mu={}".format(mu))
    # define plot types
    plots = [UCB1_notes(T, mu_star, mu, K, n_reps),
             UCB1_improved(T, mu_star, mu, K, n_reps),
             EXP3(T, mu_star, mu, K, n_reps)]
    # plot each type
    for p in plots:
        if debug: p.print_status()
        p.plot()
    # define plot parameters
    plt.xlabel("t")
    plt.ylabel("regret")
    plt.legend()
    plt.title(r"$\mu={}, K={}$".format(mu, K))
    plt.show()

def main():
    T = 10000
    mu_star = 0.5
    Ks = [2, 4, 8, 16]
    mu_diffs = [0.25, 0.125, 0.0625]
    mus = [mu_star - mu for mu in mu_diffs]
    n_reps = 10

    plot(T, mu_star, Ks[1], mus[1], n_reps)
    # x = EXP3(T, mu_star, mus[1], Ks[1], n_reps)
    # print(x.test())

    # for mu in mus:
    #     for K in Ks:
    #         plot(T, mu_star, K, mu, n_reps)

if __name__ == "__main__":
    debug = True
    main()
