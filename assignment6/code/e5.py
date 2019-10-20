import numpy as np
import matplotlib.pyplot as plt

def eta_t(K, t):
    return np.sqrt(np.log(K)/(t * K))

def empirical_regret(N_ts, deltas):
    return np.sum(N_ts * deltas)

def one_zero_with_prob(prob):
    return np.random.choice(2, p=[1-prob, prob])

def UCB1_generic(T, K, mu_star, mu, bound):
    # wind-up
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


def UCB1_plot_generic(T, K, mu_star, mu, n_reps, bound):
    acc = np.zeros(T)
    print("T =", T)
    xs = np.linspace(1, T, T, dtype=int)
    print(xs.shape)
    for n in range(n_reps):
        acc += UCB1_generic(T, K, mu_star, mu, bound)
    ys = acc/n_reps
    print(ys.shape)
    print(xs.shape)
    plt.plot(xs, ys)
    plt.show()


def UCB1_plot_notes(T, K, mu_star, mu, n_reps):
    def bound(t, N):
        return np.sqrt((3 * np.log(t))/(2 * N))
    return UCB1_plot_generic(T, K, mu_star, mu, n_reps, bound)

# def UCB1_plot_improved(T, K, mu_star, mu):
#     def bound(t, N):
#         return np.sqrt(np.log(t)/ N)
#     return UCB1_plot_generic(T, K, mu_star, mu, bound)

def EXP3(t):
    pass

def plot(T, mu_star, K, mu, n_reps):
    print("Plotting for mu={}".format(mu))
    print("UCB slides for T={}, mu_star={}, K={}, mu={}, n_reps={}". format(T, mu_star, K, mu, n_reps))
    print("UCB improved T={}, mu_star={}, K={}, mu={}, n_reps={}". format(T, mu_star, K, mu, n_reps))
    print("EXP3 T={}, mu_star={}, K={}, mu={}, n_reps={}". format(T, mu_star, K, mu, n_reps))

def main():
    T = 10000
    mu_star = 0.5
    Ks = [2, 4, 8, 16]
    mu_diffs = [0.25, 0.125, 0.0625]
    mus = [mu_star - mu for mu in mu_diffs]
    n_reps = 10

    UCB1_plot_notes(T, Ks[1], mu_star, mus[1], n_reps)
    # print(UCB1_notes(T, Ks[1], mu_star, mus[1]))
    # UCB1_improved(T, Ks[1], mu_star, mus[1])
    # for mu in mus:
    #     for K in Ks:
    #         plot(T, mu_star, K, mu, n_reps)

if __name__ == "__main__":
    debug = False
    main()
