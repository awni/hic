"""
Experiments computing the hierarchical information content (HIC) on elementary
cellular automata (ECA).
"""

import collections
import json
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os
import random
import seaborn as sns

import measures
import models
import plotting


def counts_for_size(states, h, w):
    """
    Helper function to compute the conditional count from the ECA states
    using variables of size (h, 2w) split at the midpoint in the column
    dimension, so that X and Y are of size (h, w).
    """
    cond_counts = collections.defaultdict(collections.Counter)
    rows, cols = states.shape
    dw = 2 * w
    for x in range(0, rows-h):
        for y in range(0, cols-dw):
            cells_l = tuple(states[x:x+h, y:y+w].ravel())
            cells_r = tuple(states[x:x+h, y+w:y+dw].ravel())
            cond_counts[cells_r][cells_l] += 1
    return cond_counts


def eca_hic(states, sizes):
    """
    Compute the HIC of given the ECA states using the given sizes.
    """
    mis = [measures.mutual_information(
        counts_for_size(states, *size)) for size in sizes]
    return sum((p - q)**2 for p,q in zip(mis[0:-1], mis[1:]))


def _class_results(c):
    state_size = 500
    hic_sizes = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]
    trials_per_class = 100
    results = collections.defaultdict(list)
    class_rules = [CLASS1, CLASS2, CLASS3, CLASS4][c]
    for _ in range(trials_per_class):
        # Randomly sample a rule from the class:
        rule_num = random.choice(class_rules)
        eca = models.ECA(rule_num)
        # Sample an initial state:
        states = [tuple(random.randint(0, 1) for _ in range(state_size))]
        for _ in range(2 * state_size):
            states.append(eca.udpate(states[-1]))
        states = np.array(states)[state_size:, :]
        h = eca_hic(states, hic_sizes)
        results[rn].append(h)
    return results


def compute_hic_by_class(save_load_dir, figure_path):
    """
    Plot the distribution of HIC over by class over random trials of rule and
    initial state.

    If `save_load_dir` is specified and exists will attempt to load data from it.
    Otherwise will make a new file in the directory with the results.
    """
    num_class = 4
    save_load_file = None
    if save_load_dir is not None:
        save_load_file = os.path.join(save_load_dir, "hic_by_class.json")
    if save_load_file is not None and os.path.exists(save_load_file):
        with open(save_load_file, 'r') as fid:
            results = json.load(fid)
    else:
        with mp.Pool(num_class) as pool:
            results = pool.map(_class_results, range(num_class))
    if save_load_file is not None:
        with open(save_load_file, 'w') as fid:
            json.dump(results, fid)
    all_class = []
    for c in range(num_class):
        all_class.append(
            [v for val in results[c].values() for v in val])
    results = np.array(all_class).T

    # Plots the median and a 95% confidence interval
    f, ax = plt.subplots(figsize=(10, 4))
    sns.pointplot(
        data=results, join=False, markers="d",orient="h", color="black",
        errwidth=1.0, capsize=0.2, estimator=np.median)
    ax.get_yaxis().set_ticklabels(
        ["Class {}".format(c + 1) for c in range(num_class)])
    ax.set_xlabel("HIC");
    plotting.savefig(os.path.join(args.figure_path, "hic_by_class"))


def hic_vs_num_levels(state_size, figure_path=None):
    """ HIC vs number of levels for one rule from each class. """
    num_levels = 7
    rules = [128, 2, 30, 110]

    init_state = [0]*state_size
    init_state[state_size // 2] = 1
    init_state = tuple(init_state)
    results = {}
    for rule_num in rules:
        eca = models.ECA(rule_num)
        states = [init_state]
        for _ in range(2 * state_size):
            states.append(eca.update(states[-1]))
        states = np.array(states)[state_size:, :]
        results[rule_num] = states

    hic_sizes = [(1, l) for l in range(1, num_levels + 2)]
    hics = []
    for rule_num, states in results.items():
        mis = [measures.mutual_information(
            counts_for_size(states, *size)) for size in hic_sizes]
        mi_diffs = [(p - q)**2 for p,q in zip(mis[0:-1], mis[1:])]
        hics.append(np.cumsum(mi_diffs))
    hics = np.array(hics)
    plotting.line_plot(
        hics, np.array(list(range(1, num_levels + 1))),
        xlabel="Number of levels", ylabel="HIC",
        marker=["x", "^", "o", "d"],
        linestyle=["-.", "dotted", "--", "-"],
        legend=["Rule {}".format(r) for r in rules],
        filename=os.path.join(figure_path, f"hic_vs_num_levels_size_{state_size}"))


def visualize_sample_classes(figure_path):
    """ HIC with images for one rule from each class. """
    state_size = 200
    hic_sizes = [(1, 1), (1, 2), (1, 3)]
    init_state = [0]*state_size
    init_state[state_size // 2] = 1
    init_state = tuple(init_state)
    results = {}
    for rule_num in [128, 2, 30, 110]:
        eca = models.ECA(rule_num)
        states = [init_state]
        for _ in range(2 * state_size):
            states.append(eca.update(states[-1]))
        states = np.array(states)[state_size:, :]
        results[rule_num] = (eca_hic(states, hic_sizes), states)

    # Visualize the states and show the corresponding HIC
    f, axarr = plt.subplots(2, 2, figsize=(8, 8))
    f.subplots_adjust(wspace=0.05)
    f.subplots_adjust(hspace=0.25)
    for e, result in enumerate(results.items()):
        ax = axarr[e // 2, e % 2]
        rule_num, (h, states) = result
        ax.imshow(1-states, cmap='gray')
        title = f"Class {e + 1}, Rule {rule_num}\n" + \
            "$\\textrm{HIC}(S)$" +  "= {:.2e}".format(h)
        ax.set_title(title, fontsize=14)
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
    plotting.savefig(os.path.join(figure_path, f"eca_images_and_hic"))
    plt.close(f)


if __name__ == "__main__":
    """
    Some experiments on using HIC to distinguish ECAs by the Wolfram
    classification.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description="Train a genetic neural network.")
    parser.add_argument('--save_load_dir', type=str, default=None,
        help='Path to save (or load) data from')
    parser.add_argument('--figure_path', type=str, default=".",
        help='Path to save figures to')
    args = parser.parse_args()
    visualize_sample_classes(args.figure_path)
    hic_vs_num_levels(200, args.figure_path)
    hic_vs_num_levels(500, args.figure_path)
    compute_hic_by_class(args.save_load_dir, args.figure_path)