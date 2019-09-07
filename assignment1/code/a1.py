import numpy as np

class GridWorld():
    def __init__(self, size_x, size_y, rew_mat, trans_mat, init_vals=0, policy=None):
        self.size_x = size_x
        self.size_y = size_y
        flat_size = size_x * size_y
        if isinstance(init_vals, int):
            self.Vs = np.full(flat_size, init_vals)
        else:
            self.Vs = init_vals
        self.rew_mat = rew_mat
        self.trans_mat = trans_mat
        if policy is None:
            self.policy = np.full((flat_size, 4), 1/4)
        else:
            self.policy = policy

    def __str__(self):
        def gen_line(vals):
            retval = "||"
            for v in vals:
                retval += " " + str(v) + " ||"
            return retval
        width = 2 + self.size_x * 5
        h_line = "=" * width
        retval = h_line
        for row in range(self.size_y):
            retval += "\n" + gen_line(self.Vs[row * self.size_x : (row + 1) * self.size_x])
            retval += "\n" + h_line
        return retval

    def evaluate_policy(self, theta):
        delta = 0
        while delta < theta:
            for s in range(self.size_x * self.size_y):
                # update
                v = self.Vs[s]
                self.Vs[s] = np.sum(self.policy[s] * (self.rew_mat[s] + self.Vs[self.trans_mat[s]]))
                delta = max(delta, abs(v - self.Vs[s]))

def main():
    # init_vals = np.arange(0,12)
    init_vals = 0
    actions = ["up", "down", "right", "left"]
    reward_mapping = np.array([[-1, -1, -1, -1],
                               [-1, -6, -1, -1],
                               [-1, -1, -1, -1],
                               [-1, -6, -1, -1],
                               [-1, -1, -1, -1],
                               [-1,-11, -1, -1],
                               [-1, -1, -1, -1],
                               [-6, -1,-11, -6],
                               [-1, -1, -1, -1],
                               [-6, -1, -1, -1],
                               [-1, -1, -1, -1],
                               [ 0,  0,  0,  0]])

    trans_mapping  = np.array([[0,  3,  1, 0],
                               [1,  4,  2, 0],
                               [2,  5,  2, 1],
                               [0,  6,  3, 3],
                               [1,  7,  4, 4],
                               [2,  8,  5, 5],
                               [3,  9,  7, 6],
                               [4, 10,  8, 6],
                               [5,  8,  8, 7],
                               [6,  9,  9, 9],
                               [7, 10, 11, 10],
                               [11,11, 11, 11]])

    floorplan = GridWorld(3, 4, reward_mapping, trans_mapping, init_vals)
    print(floorplan)
    floorplan.evaluate_policy(0.5)
    print(floorplan)

if __name__ == "__main__":
    main()
