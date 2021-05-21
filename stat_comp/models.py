import numpy as np


class CCA:
    """
    Cyclic Cellular Automata
    """

    def __init__(self, k, T):
        self.k = k
        self.T = T

    def update(self, state):
        """
        TODO: parameterize neighborhood (von neumann, moore)
        TODO: generalize to n-d
        TODO: parameterize radius of neighborhood (r=1, ...)

        CCA update for a Moore neighborhood.
        Params:
            state: 2D numpy array representing the state of the CCA.
            k: Number of individual cell states.
            T: Neighbor threhsold
        """
        n = state.shape[0]
        new_state = np.zeros_like(state)
        for i in range(n):
            for j in range(n):
                k_sum = 0
                c = state[i, j]
                for p in range(-1, 2):
                    for q in range(-1, 2):
                        if p == q == 0:
                            continue
                        k_sum += (state[(i+p) % n, (j+q) % n] == c)
                new_state[i, j] = (c + (k_sum >= self.T)) % self.k
        return new_state
