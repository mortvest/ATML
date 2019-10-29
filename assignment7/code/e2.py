import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


class Simulation():
    """abstract class"""
    def __init__(self, K, n_reps, color, arm_ids, rewards):
        self.T = arm_ids.shape[0]
        self.K = K
        self.n_reps = n_reps
        self.color = color
        self.arm_ids = arm_ids
        self.rewards = rewards

    def plot(self, ax, with_stds=True, with_bound=False):
        acc = []
        n_steps = min(self.T, 1000)
        xs = np.linspace(1, self.T, n_steps, dtype=int)
        for _ in range(self.n_reps):
            x = self.calc_regrets()
            acc.append(x)
        acc = np.array(acc)
        ys = np.mean(acc, axis=0)[xs-1]
        # main plot
        label = type(self).__name__
        ax.plot(xs, ys, label=label, color=self.color)
        # plot + std
        if with_stds:
            std = np.std(acc, axis=0)[xs-1]
            ax.plot(xs,
                     ys+std,
                     label="{} + STD".format(label),
                     color=self.color,
                     linestyle='dashed')
            ax.plot(xs,
                     ys-std,
                     label="{} - STD".format(label),
                     color=self.color,
                     linestyle='dashed')
        if with_bound:
            ys = self.calc_bound()[xs-1]
            ax.plot(xs,
                     ys,
                     label="bound of {}".format(label),
                     color="green",
                     linestyle='dashed')

    def print_status(self):
        print("{} for K={}".format(self.label(), self.K))

    def updated_average(avg_old, N_old, reward):
        return (avg_old * N_old + reward) / float((N_old + 1))

    def empirical_regret(N_ts, deltas):
        return np.sum(N_ts * deltas)

    def compute_cached(self, label, func):
        try:
            result = np.load(label + ".npy")
            print("loaded cached")
            return result
        except:
            result = func()
            np.save(label + ".npy", result)
            return result

    def calc_regrets(self):
        pass

    def calc_bound(self):
        pass


class UCB1(Simulation):
    def play_hand(self, t, a):
        if self.arm_ids[t-1] == a:
            return self.rewards[t-1] * self.K
        else:
            return 0

    def calc_regrets(self):
        def func():
            acc = []
            for i in range(self.K):
                acc.append(np.sum(self.rewards[self.arm_ids == i]))
            arr = np.array(acc)
            best_id = np.argsort(arr)[-1]
            best_rewards = np.copy(self.rewards)
            best_rewards[self.arm_ids!=best_id] = 0
            return np.cumsum(best_rewards) * self.K

        best_rewards_cum = self.compute_cached("best_reward", func)[:self.T]
        cum_reward = self.calc_reward(self.T, self.K)
        return best_rewards_cum - cum_reward

    def __min_arg(self, avgs, random_ties):
        if random_ties:
            min_val = np.min(avgs)
            min_args = np.argwhere(avgs == min_val).flatten()
            return np.random.choice(min_args, 1)[0]
        else:
            return np.argsort(avgs)[-1]

    def calc_reward(self, T, K):
        # initialization
        init_avg = []
        for a in range(K):
            t = a + 1
            init_avg.append(self.play_hand(t, a))

        avg_rewards = np.array(init_avg).astype(float)
        cum_reward = np.zeros(T)
        Ns = np.ones(K)
        # normal runs
        for t in range(K+1, T+1):
            # apply the bound
            bound = np.sqrt((3 * np.log(t))/(2 * Ns))
            curr_avgs = bound + avg_rewards
            # find the best hand
            best_ind = self.__min_arg(curr_avgs, False)

            # play the hand
            avg_old = avg_rewards[best_ind]
            # update times played
            N_old = Ns[best_ind]
            Ns[best_ind] += 1
            reward = self.play_hand(t, best_ind)

            # update average
            avg_rewards[best_ind] = UCB1.updated_average(avg_old, N_old, reward)
            # calculate the regret
            cum_reward[t-1] = cum_reward[t-2] + reward
        return cum_reward


class EXP3(Simulation):
    def calc_bound(self):
        def func():
            acc = []
            for t in range(1, self.T+1):
                bound = 2.0 * np.sqrt(self.K**2.0 * t * np.log(self.K))
                acc.append(bound)
            return np.array(acc)
        return self.compute_cached("bound"+str(self.K), func)

    def calc_regrets(self):
        def func():
            acc = []
            for i in range(self.K):
                acc.append(np.sum(self.rewards[self.arm_ids == i]))
            arr = np.array(acc)
            # print(arr)
            best_id = np.argsort(arr)[-1]
            # print(best_id)
            best_rewards = np.copy(self.rewards)
            best_rewards[self.arm_ids!=best_id] = 0.0
            best_rewards = 1 - best_rewards
            # return np.cumsum(best_rewards) * self.K
            return np.cumsum(best_rewards) * self.K

        # best_loss_cum = self.compute_cached("best_losses", func)[:self.T]
        best_loss_cum = func()[:self.T]
        cum_loss = self.calc_loss(self.T, self.K)
        return cum_loss - best_loss_cum

    def play_hand(self, t, a):
        if self.arm_ids[t-1] == a:
            return (1 - self.rewards[t-1]) * self.K
        else:
            return self.K

    def __eta_t(self, K, t):
        return np.sqrt(np.log(K)/(t * K**2.0))

    def calc_loss(self, T, K):
        # init data
        Ls = np.zeros(K)
        ps = np.zeros(K)
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
            # print(ps)
            # sample hand based on p
            hand = np.random.choice(ps.shape[0], p=ps)
            # print(hand)
            # play the hand
            loss = self.play_hand(t, hand)

            l_wave = loss/ps[hand]
            # update Ls
            Ls[hand] += l_wave
            Ns[hand] += 1
            regrets[t-1] = regrets[t-2] + loss
        return regrets


class Random(Simulation):
    def play_hand(self, t, a):
        if self.arm_ids[t-1] == a:
            return self.rewards[t-1] * self.K
        else:
            return 0

    def calc_regrets(self):
        def func():
            acc = []
            for i in range(self.K):
                acc.append(np.sum(self.rewards[self.arm_ids == i]))
            arr = np.array(acc)
            best_id = np.argsort(arr)[-1]
            best_rewards = np.copy(self.rewards)
            best_rewards[self.arm_ids!=best_id] = 0
            return np.cumsum(best_rewards) * self.K

        best_rewards_cum = self.compute_cached("best_reward", func)[:self.T]
        cum_reward = self.calc_reward(self.T, self.K)
        return best_rewards_cum - cum_reward

    def calc_reward(self, T, K):
        # initialization
        cum_reward = np.zeros(T)
        # normal runs
        for t in range(1, T+1):
            # choose hand randomly
            best_ind = np.random.randint(self.K)
            # play the hand
            reward = self.play_hand(t, best_ind)
            # calculate the regret
            cum_reward[t-1] = cum_reward[t-2] + reward
        return cum_reward


def load_cached(data_dir):
    print("loading", data_dir)
    try:
        return np.load(data_dir + ".npy")
    except:
        print("not cached yet, caching..")
        data = np.loadtxt(data_dir)
        np.save(data_dir + ".npy", data)
        return data


def arg_sort_rewards(K, ids, rewards):
    acc = []
    for a in range(K):
        acc.append(np.sum(rewards[ids == a]))
    return np.argsort(np.array(acc))


def remap_ids(K, ids, inds, rewards):
    flags = np.isin(ids, inds)
    new_ids = ids[flags]
    new_rewards = rewards[flags]

    remapped_ids = new_ids + K
    for new_id in range(inds.shape[0]):
        old_id = inds[new_id] + K
        remapped_ids[remapped_ids == old_id] = new_id
    return remapped_ids, new_rewards


def extract_worst_rounds(K, ids, rewards, n_worst):
    if n_worst == K-1:
        return ids, rewards
    else:
        arg_sort = arg_sort_rewards(K, ids, rewards)
        best_ind = arg_sort[-1]

        n_worst_inds = arg_sort[:n_worst]
        inds = np.insert(n_worst_inds, 0, best_ind)
        return remap_ids(K, ids, inds, rewards)


def extract_bmw_rounds(K, ids, rewards):
    arg_sort = arg_sort_rewards(K, ids, rewards)
    best_ind, median_ind, worst_ind = arg_sort[-1], arg_sort[int(K/2)], arg_sort[0]

    inds = np.array([best_ind, median_ind, worst_ind])
    return remap_ids(K, ids, inds, rewards)


def plot(K_tot, n_reps, ids, rewards, n_worst=0, bmw=False, with_bound=False, with_std=True):
    bound_tag = ""
    title_tag = ""
    if with_bound:
        bound_tag = "_bound"
        title_tag = " with bound and random strategy"
    if bmw:
        extr_ids, extr_rewards = extract_bmw_rounds(K_tot, ids, rewards)
        file_name ="plt_median" + bound_tag + ".png"
        plot_title = "Best, median, worst arms" + title_tag
        K = 3
    else:
        extr_ids, extr_rewards = extract_worst_rounds(K_tot, ids, rewards, n_worst)
        file_name = "plt_{}_worst".format(n_worst) + bound_tag + ".png"
        if n_worst == K_tot - 1:
            plot_title = "All arms" + title_tag
        else:
            plot_title = "Best + {} worst arms".format(n_worst) + title_tag
        K = n_worst + 1

    fig = plt.figure()
    ax = fig.add_subplot(111)

    print("plotting:", plot_title)
    ucb1 = UCB1(K, n_reps, "red", extr_ids, extr_rewards)
    ucb1.plot(ax, with_std)
    exp3 = EXP3(K, n_reps, "blue", extr_ids, extr_rewards)
    exp3.plot(ax, False, with_bound)

    if with_bound:
        rand = Random(K, n_reps, "yellow", extr_ids, extr_rewards)
        rand.plot(ax, with_stds=False)

    plt.xlabel("t")
    plt.ylabel("regret")
    plt.legend()
    plt.title(plot_title)
    # ax.xaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

    print("saving:", file_name)
    plt.savefig(file_name)
    plt.show()
    plt.clf()


def main():
    data_dir = "./data_preprocessed_features"
    data = load_cached(data_dir)
    ids = data[:,0].astype(int)
    click_rewards = data[:,1].astype(int)
    K = 16
    n_reps = 1

    plot(K, n_reps, ids, click_rewards, bmw=True, with_bound=True)

    # n_worst_s = [K-1,1,2,3]
    # for n_worst in n_worst_s:
    #     plot(K, n_reps, ids, click_rewards, n_worst)
    #     plot(K, n_reps, ids, click_rewards, n_worst, with_bound=True)
    # plot(K, n_reps, ids, click_rewards, bmw=True)
    # plot(K, n_reps, ids, click_rewards, bmw=True, with_bound=True)


if __name__ == "__main__":
    main()
