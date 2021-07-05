import itertools
import numpy as np


class ECA:
    """
    Elementary Cellular Automata
    We distinguish the ECA with the Wolfram class system (1 is constant, 2 is
    periodic, 3 is random, 4 is complex). The exact classes sets are from table 2
    of [2].

    [1] "Statistical Mechanics of Cellular Automata", S. Wolfram (1983),
        Review Modern Physics 55 601–644.
    [2] "A Note on Elementary Cellular Automata Classification",
        Genaro J. Martinez, 2013,  June 17, 2013,
        https://arxiv.org/abs/1306.5577
    """

    CLASS1 = [0, 8, 32, 40, 128, 136, 160, 168]
    CLASS2 = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 19, 23,
              24, 25, 26, 27, 28, 29, 33, 34, 35, 36, 37, 38, 42, 43,
              44, 46, 50, 51, 56, 57, 58, 62, 72, 73, 74, 76, 77, 78,
              94, 104, 108, 130, 132, 134, 138, 140, 142, 152, 154,
              156, 162, 164, 170, 172, 178, 184, 200, 204, 232]
    CLASS3 = [18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150]
    CLASS4 = [41, 54, 106, 110]

    def __init__(self, rule_number):
        self.rule_number = rule_number
        self.rule = self._rule_to_dict(rule_number)

    def _rule_to_dict(self, n):
        """
        Converts an elementary CA rule [0, 255] to a dictionary with states as
        keys and values representating the update for each state.
        """
        bin_rule = reversed(bin(n)[2:])
        states = itertools.product(range(2), repeat=3)
        return {x : int(y)
            for x, y in itertools.zip_longest(states, bin_rule, fillvalue=0)}

    def get_neighborhood(state, loc, radius):
        """
        Get the neighborhood of the given `radius` centered at `loc` assuming
        the state wraps around circularly.
        """
        if not (isinstance(state, list) or isinstance(state, tuple)):
            raise ValueError("State must be a list")
        neighbors = state[max(loc - radius, 0):loc + radius + 1]
        if loc - radius < 0:
            neighbors = state[loc - radius:] + neighbors
        if loc + radius >= len(state):
            ridx = loc + radius - len(state) + 1
            neighbors = neighbors + state[:ridx]
        return tuple(neighbors)

    def update(self, state):
        """
        Updates a state description of a 1D ECA state using the dictionary
        representing the provided rule. We assume the state is connected at the
        boundaries to form a ring.
        """
        n = len(list(self.rule.keys())[0])
        radius = n // 2
        n_cells = len(state)
        state = state[-radius:] + state + state[:radius]
        return tuple(self.rule[state[i:i+n]] for i in range(n_cells))


class CCA:
    """
    Cyclic Cellular Automata
    """

    def __init__(self, k, T):
        self.k = k
        self.T = T


    def get_neighborhood(state, location, radius):
        i, j = location
        return tuple(state[(i+p) % n, (j+q) % n]
                    for p in range(-radius, radius+1)
                        for q in range(-radius, radius+1))


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
