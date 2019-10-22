import numpy as np
import matplotlib.pyplot as plt


class Plot():
    """abstract class"""
    def __init__(self, T, mu_star, mu, K, n_reps, color, adv_seq=None):
        self.T = T
        self.mu_star = mu_star
        self.mu = mu
        self.K = K
        self.n_reps = n_reps
        self.color = color
        self.adv_seq=adv_seq

    def label(self):
        pass

    def plot(self, with_stds=True):
        acc = np.zeros(self.T)
        xs = np.linspace(1, self.T, self.T, dtype=int)
        for n in range(self.n_reps):
            x = self.calc_regrets(self.T, self.K, self.mu_star, self.mu, self.adv_seq)
            acc = np.vstack((acc, x))
        acc = acc[1:]
        ys = np.mean(acc, axis=0)
        # main plot
        plt.plot(xs, ys, label=self.label(), color=self.color)
        if with_stds:
            std = np.std(acc, axis=0)
            plt.plot(xs,
                     ys+std,
                     label="{} + STD".format(self.label()),
                     color=self.color,
                     linestyle='dashed')

    def plot_adv(self):
        n = self.T-self.K
        acc = np.zeros(n)
        xs = np.linspace(self.K, self.T, n, dtype=int)
        for n in range(self.n_reps):
            x = self.calc_regrets(self.T, self.K, self.mu_star, self.mu, self.adv_seq)
            acc = np.vstack((acc, x))
        acc = acc[1:]
        best_column_ind = np.argsort(np.sum(self.adv_seq, axis=0))[-1]
        best_column = self.adv_seq[:,best_column_ind]
        cumsum = np.cumsum(best_column)

        print(acc[-1,-1])
        print(cumsum[-1])
        diff = np.subtract(cumsum, acc)
        ys = np.mean(diff, axis=0)
        plt.plot(xs, ys, label=self.label(), color=self.color)


    def calc_regrets(self, T, K, mu_star, mu):
        """implement for each type of plot"""
        pass

    def print_status(self):
        print("{} for K={}, mu={}".format(self.label(), self.K, self.mu))


class UCB1(Plot):
    """generic UCB1"""
    def calc_regrets(self, T, K, mu_star, mu, adv_seq):
        return self.__UCB1_generic(T, K, mu_star, mu, self.bound, adv_seq)
    def bound(self, t, Ns):
        """implement for each type of UCB1"""
        pass
    def __UCB1_generic(self, T, K, mu_star, mu, bound, adv_seq):
        # initialization
        if adv_seq is None:
            init_avg = [one_zero_with_prob(mu_star)]
            for _ in range(K-1):
                init_avg.append(one_zero_with_prob(mu))
            avg_rewards = np.array(init_avg).astype(float)
            probs = np.array([mu_star] + [mu] * (K-1))
        else:
            avg_rewards = np.array([1] + [0]*(K-1)).astype(float)

        regrets = np.zeros(T-K)
        Ns = np.ones(K)
        # normal runs
        for t in range(K+1, T+1):
            # apply the bound
            # print("Ns", Ns)
            # print("bound", bound(t,Ns))
            curr_avgs = bound(t, Ns) + avg_rewards
            # find the best hand
            best_ind = np.argsort(curr_avgs)[-1]
            # play the hand
            avg_old = avg_rewards[best_ind]
            # update times played
            N_old = Ns[best_ind]
            Ns[best_ind] += 1
            if adv_seq is None:
                gain = one_zero_with_prob(probs[best_ind])
                # update average
                avg_rewards[best_ind] = (avg_old * N_old + gain) / float((N_old + 1))
                # calculate the regret
                emp_reg = empirical_regret(Ns, (mu_star - probs))
                regrets[t-1] = emp_reg
            else:
                # print("Playing", best_ind)
                gain = adv_seq[t-K-1, best_ind]
                # print("t", t)
                # print("i", t-K-1)

                # print("v",adv_seq[t-K-1])
                # update average
                avg_rewards[best_ind] = (avg_old * N_old + gain) / float((N_old + 1))
                # calculate the regret
                # adv_reg = gain - np.min(adv_seq[t-K-1])
                adv_reg = gain
                if t == 1:
                    regrets[0] = adv_reg
                else:
                    regrets[t-K-1] = regrets[t-K-2] + adv_reg
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
        return self.calc_regrets(self.T, self.K, self.mu_star, self.mu, self.adv_seq)
    def calc_regrets(self, T, K, mu_star, mu, adv_seq):
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


def plot(T, mu_star, K, mu, n_reps, with_stds=True):
    if debug: print("Plotting for mu={}, K={}".format(mu, K))
    # define plot types
    plots = [UCB1_notes(T, mu_star, mu, K, n_reps, color="red"),
             UCB1_improved(T, mu_star, mu, K, n_reps, color="blue"),
             EXP3(T, mu_star, mu, K, n_reps, color="green")]
    # plot each type
    for p in plots:
        if debug: p.print_status()
        p.plot(with_stds)
    # define plot parameters
    plt.xlabel("t")
    plt.ylabel("regret")
    plt.legend()
    plt.title(r"$\mu={}, K={}$".format(mu, K))
    # plt.savefig("plt_k{}_mu{}".format(K, mu).replace(".","") + ".png")
    plt.show()
    plt.clf()

def plot_adv(T, K, n_reps, adv_seq):
    if debug: print("Plotting ADV for K={}".format(K))
    # define plot types
    # plots = [UCB1_notes(T, mu_star, mu, K, n_reps, "red", adv_seq),
    #          EXP3(T, mu_star, mu, K, n_reps, "green", adv_seq)]
    plots = [UCB1_notes(T, 0, 0, K, n_reps, "red", adv_seq)]
    # plot each type
    for p in plots:
        if debug: p.print_status()
        p.plot_adv()
    # define plot parameters
    plt.xlabel("t")
    plt.ylabel("regret")
    plt.legend()
    plt.title(r"Adversarial sequence".format(K))
    # plt.savefig("plt_adv.png")
    plt.show()
    plt.clf()


def create_adv_seq(T, K):
    def bound(t,N): return np.sqrt((3 * np.log(t))/(2 * N))
    # initialize hands
    seq = np.zeros(K)
    avg_rewards = np.array([1] + [0]*(K-1)).astype(float)
    Ns = np.ones(K)
    regrets = np.zeros(T-K)
    # start building adversarial sequence
    for t in range(K+1, T+1):
        # print("Ns", Ns)
        # apply the bound
        curr_avgs = bound(t, Ns) + avg_rewards
        # find the best hand
        best_ind = np.argsort(curr_avgs)[-1]
        # print("curr_avgs_", curr_avgs)
        worst_ind = np.argsort(curr_avgs)[0]
        # play the hand
        avg_old = avg_rewards[best_ind]
        # update times played
        N_old = Ns[best_ind]
        Ns[best_ind] += 1
        # update average
        avg_rewards[best_ind] = (avg_old * N_old) / float((N_old + 1))
        row = np.zeros(K)
        row[worst_ind] = 1
        seq = np.vstack((seq, row))

        # print("t", t)
        # print("v", row)
        # print("v", curr_avgs)
        # print("worst ind", worst_ind)
        # print("best ind", best_ind)
        # print("row", row)
        # adv_reg = seq[T-K-1, best_ind]
        # if t == 1:
        #     print("taken")
        #     regrets[0] = adv_reg
        # else:
        #     regrets[t-K-1] = regrets[t-K-2] + adv_reg
    seq = seq[1:]
    return seq

def main():
    T = 10000
    mu_star = 0.5
    Ks = [2, 4, 8, 16]
    mu_diffs = [0.25, 0.125, 0.0625]
    mus = [mu_star - mu for mu in mu_diffs]
    # n_reps = 10
    n_reps = 1
    # T = 10
    # mu_star = 0.5
    # Ks = [2, 4, 8, 16]
    # mu_diffs = [0.25, 0.125, 0.0625]
    # mus = [mu_star - mu for mu in mu_diffs]
    # n_reps = 1


    adv_seq = create_adv_seq(T, Ks[0])
    print("adv_seq.shape", adv_seq.shape)

    plot_adv(T, Ks[0], n_reps, adv_seq)
    # plot(T, mu_star, Ks[1], mus[1], n_reps)
    # x = EXP3(T, mu_star, mus[1], Ks[1], n_reps, "red")
    # x.plot()
    # for mu in mus:
    #     for K in Ks:
    #         plot(T, mu_star, K, mu, n_reps)


if __name__ == "__main__":
    debug = True
    main()
