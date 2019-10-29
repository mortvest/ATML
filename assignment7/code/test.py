import numpy as np

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

def remap_ids(K, ids, inds):
    new_ids = ids + K
    for new_id in range(inds.shape[0]):
        old_id = inds[new_id] + K
        new_ids[new_ids == old_id] = new_id
    return new_ids

def extract_worst_rounds(K, ids, rewards, n_worst):
    if n_worst == K-1:
        return ids, rewards
    else:
        arg_sort = arg_sort_rewards(K, ids, rewards)
        best_ind = arg_sort[-1]

        n_worst_inds = arg_sort[:n_worst]
        inds = np.insert(n_worst_inds, 0, best_ind)
        flags = np.isin(ids, inds)

        new_ids = ids[flags]
        new_rewards = rewards[flags]

        remapped_ids = remap_ids(K, new_ids, inds)
        return remapped_ids, new_rewards

def extract_bmw_rounds(K, ids, rewards):
    arg_sort = arg_sort_rewards(K, ids, rewards)
    best_ind, median_ind, worst_ind = arg_sort[-1], arg_sort[int(K/2)], arg_sort[0]

    inds = np.array([best_ind, median_ind, worst_ind])
    flags = np.isin(ids, inds)

    new_ids = ids[flags]
    new_rewards = rewards[flags]

    remapped_ids = remap_ids(K, new_ids, inds)
    return remapped_ids, new_rewards

def main():
    data_dir = "./data_preprocessed_features"
    data = load_cached(data_dir)
    ids = data[:,0].astype(int)
    click_rewards = data[:,1].astype(int)
    K = 16
    # inds, extr_rewards = extract_worst_rounds(K, ids, click_rewards, 3)
    inds, extr_rewards = extract_bmw_rounds(K, ids, click_rewards)
    print(inds[:20], extr_rewards[:20])

if __name__ == "__main__":
    main()
