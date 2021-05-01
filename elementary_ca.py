import collections
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import seaborn as sns

import plotting

"""
 Experiments computing the hierarchical information content (HIC) of elementary
 cellular automata (ECA).

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


def rule_to_dict(n):
    """
    Converts an elementary CA rule [0, 255] to a dictionary with states as
    keys and values representating the update for each state.
    """
    bin_rule = reversed(bin(n)[2:])
    states = itertools.product(range(2), repeat=3)
    return {x : int(y)
        for x, y in itertools.zip_longest(states, bin_rule, fillvalue=0)}


def update(state, rule):
    """
    Updates a state description of a 1D CA state using the dictionary representing the
    provided rule. We assume the state is connected at the boundaries to form a ring.
    """
    n = len(list(rule.keys())[0])
    radius = n // 2
    n_cells = len(state)
    state = state[-radius:] + state + state[:radius]
    return tuple(rule[state[i:i+n]] for i in range(n_cells))


def entropy(counts):
    """
    Compute the Shannon entropy of a multinomial distribution given counts in a
    iterable or a Counter.
    """
    total = 0
    inds = 0
    if isinstance(counts, collections.Counter):
        counts = (c for _, c in counts.most_common())
    for c in counts:
        total += c
        inds += c * math.log(c)
    entropy = math.log(total) - (1 / total) * inds
    return entropy, total


def mutual_information(states, h, w):
    """
    Compute the mutual information I(X; Y) from the ECA states using variables
    of size (h, 2w) split at the midpoint in the column dimension, so that X
    and Y are of size (h, w)
    """
    cond_counts = collections.defaultdict(collections.Counter)
    rows, cols = states.shape
    dw = 2 * w
    for x in range(0, rows-h):
        for y in range(0, cols-dw):
            cells_l = tuple(states[x:x+h, y:y+w].ravel())
            cells_r = tuple(states[x:x+h, y+w:y+dw].ravel())
            cond_counts[cells_r][cells_l] += 1

    # First compute the conditional entropy: sum_y p(y) sum_x p(x|y) log 1/p(x|y)
    cond_ents = [entropy(xs) for xs in cond_counts.values()]
    cond_ent_sum = sum(cond_ent_y*ny for cond_ent_y, ny in cond_ents)
    y_tot = sum(ny for _, ny in cond_ents)
    cond_ent = cond_ent_sum / y_tot

    # Then compute the entropy: sum_y p(x) log 1 / p(x)
    # Since the x's and ys are essentially the same, we can
    # use the y counts
    ent = entropy(list(ny for _, ny in cond_ents))[0]

    # Return the mutual information I(x, y) = H(x) - H(x|y)
    return ent - cond_ent


def hic(states, sizes):
    """
    Compute the HIC of states using the given sizes.
    """
    mis = [mutual_information(states, *size) for size in sizes]
    return sum((p - q)**2 for p,q in zip(mis[0:-1], mis[1:]))


def compute_hic_by_class():
    """
    Plot the distribution of HIC over by class over random trials of rule and
    initial state.
    """
    classes = [CLASS1, CLASS2, CLASS3, CLASS4]
    state_size = 200
    hic_sizes = [(1, 1), (1, 2), (1, 3)]
    trials_per_class = 10
    results = collections.defaultdict(list)
    for c in range(4):
        for t in range(trials_per_class):
            # sample a rule:
            rn = random.choice(classes[c])
            rule = rule_to_dict(rn)
            # sample an initial state:
            states = [tuple(random.randint(0, 1) for _ in range(state_size))]
            for _ in range(2 * state_size):
                states.append(update(states[-1], rule))
            states = np.array(states)[state_size:, :]
            h = hic(states, hic_sizes)
            print(c, rn, h)
            results[c].append(h)

    f, ax = plt.subplots(figsize=(10, 4))
    results = np.array([results[c] for c in range(4)]).T
    sns.stripplot(data=results, alpha=.25, zorder=1, orient="h", color="black")
    sns.pointplot(
        data=results, join=False, markers="d", ci="sd",
        errwidth=0.5, capsize=0.1, orient="h", color="black")
    ax.get_yaxis().set_ticklabels(["Class {}".format(c) for c in range(1, 5)])
    ax.set_xlabel("HIC")
    plotting.savefig(os.path.join("paper/figures", "hic_by_class"))

def hic_vs_num_levels(state_size=200):
    """ HIC vs number of levels for one rule from each class. """
    num_levels = 7
    rules = [128, 2, 30, 110]

    init_state = [0]*state_size
    init_state[state_size // 2] = 1
    init_state = tuple(init_state)
    results = {}
    for rule_num in rules:
        rule = rule_to_dict(rule_num)
        states = [init_state]
        for _ in range(2 * state_size):
            states.append(update(states[-1], rule))
        states = np.array(states)[state_size:, :]
        results[rule_num] = states

    hic_sizes = [(1, l) for l in range(1, num_levels + 2)]
    hics = []
    for rule_num, states in results.items():
        mis = [mutual_information(states, *size) for size in hic_sizes]
        mi_diffs = [(p - q)**2 for p,q in zip(mis[0:-1], mis[1:])]
        hics.append(np.cumsum(mi_diffs))
    hics = np.array(hics)
    plotting.line_plot(
        hics, np.array(list(range(1, num_levels + 1))),
        xlabel="Number of levels", ylabel="HIC",
        legend=["Rule {}".format(r) for r in rules],
        filename=os.path.join("paper/figures/", f"hic_vs_num_levels_size_{state_size}"))


def visualize_sample_classes():
    """ HIC with images for one rule from each class. """
    state_size = 200
    hic_sizes = [(1, 1), (1, 2), (1, 3)]
    init_state = [0]*state_size
    init_state[state_size // 2] = 1
    init_state = tuple(init_state)
    results = {}
    for rule_num in [128, 2, 30, 110]:
        rule = rule_to_dict(rule_num)
        states = [init_state]
        for _ in range(2 * state_size):
            states.append(update(states[-1], rule))
        states = np.array(states)[state_size:, :]
        results[rule_num] = (hic(states, hic_sizes), states)

    # Visualize the states and show the corresponding HIC
    f, axarr = plt.subplots(2, 2, figsize=(8, 8))
    f.subplots_adjust(wspace=0.05)
    for e, result in enumerate(results.items()):
        ax = axarr[e // 2, e % 2]
        rule_num, (h, states) = result
        ax.imshow(1-states, cmap='gray')
        title = f"Rule {rule_num}; " + "$\\textrm{HIC}(S)$" +  "= {:.2e}".format(h)
        ax.set_title(title, fontsize=14)
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
    plotting.savefig(os.path.join("paper/figures", f"eca_images_and_hic"))
    plt.close(f)


if __name__ == "__main__":
    """
    Some experiments on using HIC to distinguish ECAs by the Wolfram
    classification.
    """
    #visualize_sample_classes()
    #hic_vs_num_levels(200)
    #hic_vs_num_levels(500)
    compute_hic_by_class()
