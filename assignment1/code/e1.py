import numpy as np

class GridWorld():
    """ Grid world object

    Args:
        size_x (int): first dimension of the world
        size_y (int): second dimension of the world
        rew_mat (numpy array): matrix of rewards
        trans_mat (numpy array): transition matrix
        init_vals (int or np.array): initial values for the Vs
        policy (numpy array): policy matrix, set to random by default
    """
    def __init__(self, size_x, size_y, rew_mat, trans_mat, init_vals=None, policy=None):
        self.size_x = size_x
        self.size_y = size_y
        flat_size = size_x * size_y
        if policy is None:
            self.Vs = np.full(flat_size, 0.0)
        else:
            self.Vs = init_vals
        self.rew_mat = rew_mat
        self.trans_mat = trans_mat
        if policy is None:
            self.policy = np.full((flat_size, 4), 1/4)
        else:
            self.policy = policy

    def __str__(self):
        return str(self.Vs.reshape(self.size_y, self.size_x).round(3))

    def evaluate_policy(self, theta):
        """
        Evaluates policy using dynamic programming
        Args:
            theta (float): threshold value
        """
        if debug: print("Evaluating policy")
        while True:
            delta = 0
            for s in range(self.size_x * self.size_y):
                # update
                v = self.Vs[s]
                self.Vs[s] = np.sum(self.policy[s] * (self.rew_mat[s] + self.Vs[self.trans_mat[s]]))
                delta = max(delta, abs(v - self.Vs[s]))
            if delta < theta:
                break

    def iterate_policy(self, theta):
        """
        Iterates policy using dynamic programming
        Args:
            theta (float): threshold value for the policy evaluation
        """
        def find_best_action(s):
            # TODO: check this
            diag_mat = np.diag(np.ones(4))
            diag_mult = diag_mat * (self.rew_mat[s] + self.Vs[self.trans_mat[s]])
            if debug: print(diag_mult)
            arr = np.diag(diag_mult)
            if debug: print(arr)
            max_ind = np.argmax(arr)
            if debug: print(max_ind)
            res_policy = diag_mat[max_ind]
            if debug: print(res_policy)
            return res_policy

        f_stable = False
        while not f_stable:
            if debug: print("Iterating")
            # policy evaluation
            self.evaluate_policy(theta)
            f_stable = True
            # policy improvement
            for s in range(self.size_x * self.size_y):
                a = np.copy(self.policy[s])
                self.policy[s] = find_best_action(s)
                if not np.array_equal(a, self.policy[s]):
                    f_stable = False

def main():
    ## SETUP
    # setup the transition and reward mappings
    init_vals = 0
    theta = 0.1
    # actions = ["up", "down", "right", "left"]
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

    print("TASK 1")

    # create a 3x4 grid world object with a random policy
    floorplan = GridWorld(3, 4, reward_mapping, trans_mapping, init_vals)

    # evaluate policy
    floorplan.evaluate_policy(theta)

    # view the resulting 12 values of Vrand
    print("Value function of a random policy")
    print(floorplan)


    print("")
    print("TASK 2")

    # create a 3x4 grid world object with a random default policy
    floorplan = GridWorld(3, 4, reward_mapping, trans_mapping, init_vals)
    print("Starting policy:")
    print(floorplan.policy)

    # iterate policy
    floorplan.iterate_policy(theta)
    print("Optimal policy:")
    print(floorplan.policy)

    # view the resulting 12 values of V*
    print("Optimal value function:")
    print(floorplan)


if __name__ == "__main__":
    debug = False
    assert(True)
    main()
