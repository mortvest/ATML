import numpy as np
import matplotlib.pyplot as plt


class Simulation():
    """abstract class"""
    def __init__(self, T, K, n_reps, color, arm_ids, rewards):
        self.T = T
        self.K = K
        self.n_reps = n_reps
        self.color = color
        self.arm_ids = arm_ids
        self.rewards = rewards

    def plot(self, with_stds=True):
        acc = []
        xs = np.linspace(self.K, self.T, self.T, dtype=int)
        for _ in range(self.n_reps):
            x = self.calc_regrets()
            acc.append(x)
        acc = np.array(acc)
        ys = np.mean(acc, axis=0)
        # main plot
        label = type(self).__name__
        plt.plot(xs, ys, label=label, color=self.color)
        # plot + std
        if with_stds:
            std = np.std(acc, axis=0)
            plt.plot(xs,
                     ys+std,
                     label="{} + STD".format(label),
                     color=self.color,
                     linestyle='dashed')


    def calc_regrets(self):
        pass

    def print_status(self):
        print("{} for K={}".format(self.label(), self.K))

    def plot_bound(self, T, K):
        """implement for each type of simulation"""
        pass

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
        # best_rewards_cum = func()[:self.T]
        cum_reward = self.calc_reward(self.T, self.K)
        return best_rewards_cum - cum_reward

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
            best_ind = np.argsort(curr_avgs)[-1]
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
    def play_hand(self, t, a):
        if self.arm_ids[t-1] == a:
            # return (1 - self.rewards[t-1]) * 16
            return (1 - self.rewards[t-1]) 
        else:
            return 1

    def __eta_t(self, K, t):
        return np.sqrt(np.log(K)/(t * K))

    def calc_regrets(self):
        cum_loss = self.calc_loss(self.T, self.K)
        best_loss = 1 - np.cumsum(self.rewards[:self.T])
        return cum_loss - best_loss

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
            # hand = np.random.choice(ps.shape[0], p=ps)
            hand = np.random.randint(self.K)
            # play the hand
            loss = self.play_hand(t, hand)

            l_wave = loss/ps[hand]
            # update Ls
            Ls[hand] += l_wave
            Ns[hand] += 1
            regrets[t-2] = regrets[t-1] + loss
        return regrets



def load_cached(data_dir):
    print("loading", data_dir)
    try:
        return np.load(data_dir + ".npy")
    except:
        print("not cached yet, caching..")
        data = np.loadtxt(data_dir)
        np.save(data_dir + ".npy", data)
        return data


def extract_rounds(K, ids, rewards, n_worst):
    # TODO: replace with 0...n
    acc = []
    for a in range(K):
        acc.append(np.sum(rewards[ids == a]))
    arg_sort = np.argsort(np.array(acc))
    best_ind = arg_sort[-1]
    n_worst = arg_sort[:n_worst]
    inds = np.insert(n_worst, 0, best_ind)
    flags = np.isin(ids, inds)

    new_ids = ids[flags]
    new_rewards = rewards[flags]
    return new_ids, new_rewards


def plot(T, K, n_reps, ids, rewards, n_worst, with_std=False):
    extr_ids, extr_rewards = extract_rounds(K, ids, rewards, n_worst)
    ucb1 = UCB1(T, K, n_reps, "red", extr_ids, extr_rewards)
    ucb1.plot(with_std)
    # exp3 = EXP3(T, K, n_reps, "blue", extr_ids, extr_rewards)
    # exp3.plot(with_std)
    plt.xlabel("t")
    plt.ylabel("regret")
    plt.legend()
    # plt.savefig("plt_{}_worst.png".format(n_worst))
    plt.show()
    plt.clf()

# def plot_v(T, K, n_reps, ids, rewards, with_stds=False):
#     extr_ids, extr_rewards = extract_bmv(K, ids, rewards, n_worst)
#     ucb1 = UCB1(T, K, n_reps, "red", extr_ids, extr_rewards)
#     ucb1.plot(with_std)
#     exp3 = EXP3(T, K, n_reps, "blue", extr_ids, extr_rewards)
#     exp3.plot(with_std)
#     plt.xlabel("t")
#     plt.ylabel("regret")
#     plt.legend()
#     plt.savefig("plt_median.png".format(n_worst))
#     # plt.show()
#     plt.clf()

# def copute_cached()

def main():
    data_dir = "./data_preprocessed_features"
    data = load_cached(data_dir)
    ids = data[:,0].astype(int)
    click_rewards = data[:,1].astype(int)
    T = ids.shape[0]
    # T = 2000
    K = 16
    n_reps = 1
    # n_reps = 1
    n_worst_s = [K-1,1,2,3]
    inds, extr_rewards = extract_rounds(K, ids, click_rewards, 1)

    # plot(T, K, n_reps, ids, click_rewards, K-1)

    # for n_worst in n_worst_s:
    #     plot(T, K, n_reps, ids, click_rewards, n_worst)

if __name__ == "__main__":
    main()
