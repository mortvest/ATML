import numpy as np
import matplotlib.pyplot as plt

def eta_t(K, t):
    return np.sqrt(np.log(K)/(t * K))

def pseudo_regret(N_ts, deltas):
    return np.sum(N_ts * deltas)

def one_zero_with_prob(prob):
    return np.random.choice(2, p=[1-prob, prob])


def UCB1_notes(T, K, mu_star, mu):
    def bound(t, N):
        return np.sqrt((3 * np.log(t))/(2 * N))
    def single_round(t):
        # apply the bound
        curr_avgs = bound(t, Ns) + avg_rewards
        # find the best hand
        best_ind = np.argsort(curr_avgs)[-1]
        # play the hand
        if debug: print("playing hand #{}".format(best_ind))
        gain = one_zero_with_prob(probs[best_ind])
        # update average
        N_old = Ns[best_ind]
        avg_old = avg_rewards[best_ind]
        if debug: print("Before update", avg_rewards)
        avg_rewards[best_ind] = (avg_old * N_old + gain) / float((N_old + 1))
        if debug: print("value", (avg_old * N_old + gain) / ((N_old + 1)))
        if debug: print("After update", avg_rewards)
        Ns[best_ind] += 1

    # wind-up
    init_avg = [one_zero_with_prob(mu_star)]
    for _ in range(K-1):
        init_avg.append(one_zero_with_prob(mu))
    avg_rewards = np.array(init_avg).astype(float)
    probs = np.array([mu_star] + [mu] * (K-1))
    if debug: print("Probs", probs)
    Ns = np.ones(K)
    # normal runs
    if debug: print("Init avgs", avg_rewards)
    for t in range(1, int(T)+1):
    # for t in range(1, 10):
        single_round(t)
    # if debug: print("After rounds:", avg_rewards)
    print("After rounds:", avg_rewards)


def UCB1_imporoved(t):
    pass

def EXP3(t):
    pass

def plot(T, mu_star, K, mu, n_reps):
    # for _ in n_reps:
    if debug:
        print("Plotting for T={}, mu_star={}, K={}, mu={}, n_reps={}".
              format(T, mu_star, K, mu, n_reps))

def main():
    T = 10000.
    mu_star = 0.5
    Ks = [2, 4, 8, 16]
    mu_diffs = [0.25, 0.125, 0.0625]
    mus = [mu_star - mu for mu in mu_diffs]
    n_reps = 10

    UCB1_notes(T, Ks[1], mu_star, mus[1])
    # for K in Ks:
    #     for mu in mus:
    #         plot(T, mu_star, K, mu, n_reps)

if __name__ == "__main__":
    debug = False
    main()
