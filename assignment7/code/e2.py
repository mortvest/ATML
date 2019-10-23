import numpy as np
import matplotlib.pyplot as plt

def load_cached(data_dir):
    print("loading", data_dir)
    try:
        return np.load(data_dir + ".npy")
    except:
        print("not cached yet, caching..")
        data = np.loadtxt(data_dir)
        np.save(data_dir + ".npy", data)
        return data

def main():
    data_dir = "./data_preprocessed_features"
    data = load_cached(data_dir)
    ids = data[:,0].astype(int)
    click_inds = data[:,1].astype(int)
    # print("First 10 ids:", ids[:10])
    # print("First 10 click indicators:", click_inds[:10])
    n_clicks = np.sum(click_inds)
    print("n times clicked:", n_clicks)

if __name__ == "__main__":
    main()
