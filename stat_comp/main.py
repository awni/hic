import argparse
import collections
import numpy as np
import random

import distances
import models

def get_neighborhood(state, location, radius):
    i, j = location
    return tuple(state[(i+p) % n, (j+q) % n]
                for p in range(-radius, radius+1)
                    for q in range(-radius, radius+1))


def get_past_cone(prev_states, location):
    t = len(prev_states)
    return tuple(get_neighborhood(state, location, t - i - 1)
                     for i, state in enumerate(prev_states))


def get_future_cone(future_states, location):
    return tuple(get_neighborhood(state, location, i+1)
                     for i, state in enumerate(future_states))



def generate_causal_states(future_given_past, distance_metric, threshold):
    """
    Causal states are generated by merging past cones with "similar"
    distributions over future cones.

    Merging can have a range of implementations. Virtually any clustering
    algorithm which can infer the number of clusters will work. Here we do
    something simple:
        For each cone in a random order over past cones:
        1. Compute distances from the next state's conditional probabilities
           to those of the already constructed causal states.
        2. Merge with the closest if the distance is below `threshold`,
           otherwise construct a new causal state.

    future_given_past: should be a dictionary key'ed by the past light cone states.
        Each value is the corresponding conditional distribution over future
        light-cone states which can be represented in an arbitrary way subject
        to the following constraints:
            1. Compatable with the user-defined distance_metric.
            2. The conditional distributions must support an `update` operation.
    distance_metric: takes two future light cone conditional distributions and returns
        a scalar distance. This need not be a proper metric.
    threshold: A scalar threshold on the distance required to merge two conditional
        distributions. This threshold should be the same when comparing statistical
        complexity of two different processes as it dictates the number of causal states.
    """
    past_states = list(future_given_past.keys())
    random.shuffle(past_states)

    # Keeps track of the conditional probability distribution for each causal
    # state as a list of counts
    causal_states = [future_given_past[past_states[0]].copy()]
    for past_cone in past_states[1:]:
        # Compute distances to already constructed causal states
        dists = [distance_metric(future_given_past[past_cone], cond_probs)
                    for cond_probs in causal_states]
        # Find the argmin
        argmin, dist = min(enumerate(dists), key = lambda pair: pair[1])
        if dist < threshold:
            # Merge when the distance is less than the threshold
            causal_states[argmin].update(future_given_past[past_cone])
        else:
            # Otherwise create a new causal state
            causal_states.append(future_given_past[past_cone].copy())

    return causal_states


def compute_local_statistical_complexity(past_states, future_states):
    """
    Compute the statistical complexity after a single update.
        past_states is a list of states
        future_states is a list of states
    """
    future_given_past = collections.defaultdict(collections.Counter)

    ## Make a map of past light cones to a map of future light cones with counts
    for i in range(n):
        for j in range(n):
            # get the past light cone at (i, j)
            past_cone = get_past_cone(past_states, (i, j))
            # get the corresponding  future light cone
            future_cone = get_future_cone(future_states, (i, j))
            # update conditional probability table
            future_given_past[past_cone][future_cone] += 1

    # Generate the set of causal states
    causal_states = generate_causal_states(future_given_past, distances.l2, 0.1)

    # Compute the entropy of the distribution of causal states
    counts = [sum(t for _, t in c.most_common()) for c in causal_states]
    return distances.multinomial_entropy(counts)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the statistical complexity computation")
    parser.add_argument("--seed",  type=int, default=2021,
        help="Manual seed for RNGs")
    args = parser.parse_args()

    n = 100 # 2D Grid size
    k = 4 # Number of colors
    T = 2 # Update threshold
    dp = 2 # Past light-cone depth (includes current state)
    df = 1 # Future light-cone depth

    random.seed(args.seed)
    np.random.seed(args.seed + 1)
    state = np.random.randint(low=0, high=k, size=(n, n))
    cca = models.CCA(k, T)
    states = [state]
    for _ in range(10):
        states.append(cca.update(states[-1]))

    for i in range(1, len(states) - df + 1):
        past_states = states[max(i - dp, 0):i] # includes current state
        future_states = states[i:i+df]
        sc = compute_local_statistical_complexity(past_states, future_states)
        print(f"Iterations {i}: Statistical Complexity {sc}")