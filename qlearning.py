import numpy as np

class QLearning:
    def __init__(self, p):
        self.R = p.get_R()
        self.T = p.get_T()

    def bellman_backup(self, U, s, gamma=1):
        maximum = float('-inf')
        for a in range(134):
            maximum = max(maximum, self.R[s, a] + gamma * np.sum([self.T[s, a, sp]*U[sp] for sp in range(16)]))
        return maximum

    def value_iteration(self):
        U = np.zeros(16)
        for k in range(1000):
            U = np.array([self.bellman_backup(U, s) for s in range(16)])
        return U

    def initialize_q(self, gamma=1):
        # Run value iteration first to get an initial value for U over the whole dataset.
        U = self.value_iteration()  # ignore the last four elements of this array

        # Initialize Q for Q learning
        Q = np.zeros((16, 134))

        for s in range(16):
            for a in range(134):
                Q[s,a] = self.R[s, a] + np.sum([self.T[s, a, sp]*U[sp] for sp in range(16)])

        return Q

    def extract_policy(self, Q, actions):
        pi = np.zeros(12)

        for s in range(12):
            pi[s] = max(range(len(Q[s])), key=Q[s].__getitem__)
        return [actions[int(pi[i])] for i in range(12)]

    def QLearn(self, Q, obs, eta, gamma=1):
        for i in range(100):
            for _, row in obs.iterrows():
                s, a, r, sp = int(row['s']), int(row['a']), row['r'], int(row['sp'])
                Q[s, a] = Q[s, a] + eta*(r + gamma*max(Q[sp,:]) - Q[s, a]) # Update Q
        return Q