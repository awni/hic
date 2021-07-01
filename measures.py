import collections
import numpy as np
import math


def _ent_impl(counts):
    total = 0
    inds = 0
    if isinstance(counts, collections.Counter):
        counts = (c for _, c in counts.most_common())
    for c in counts:
        total += c
        if c == 0:
            continue
        inds += c * math.log(c)
    entropy = math.log(total) - (1 / total) * inds
    return entropy, total


def entropy(counts):
    """
    Compute the Shannon entropy of a multinomial distribution given counts in a
    iterable or a Counter
        H(x) = \sum_x p(x) \log (1 / p(x))
    """
    return _ent_impl(counts)[0]


def conditional_entropy(cond_counts):
    """
    Compute the conditional entropy of a conditional multinomial distribution
    given a list of lists of counts or Counters for x given each y:
        H(x|y) = \sum_y p(y) \sum_x p(x|y) \log (1 / p(x|y))
    """
    if isinstance(cond_counts, dict):
        cond_counts = list(cond_counts.values())
    cond_ents = [_ent_impl(xs) for xs in cond_counts]
    cond_ent_sum = sum(cond_ent_y * ny for cond_ent_y, ny in cond_ents)
    y_tot = sum(ny for _, ny in cond_ents)
    return cond_ent_sum / y_tot


def mutual_information(cond_counts):
    """
    Compute the mutual information I(X; Y):
        I(x, y) = H(x) - H(x|y)
    """
    if isinstance(cond_counts, dict):
        cond_counts = list(cond_counts.values())
    if isinstance(cond_counts[0], collections.Counter):
        cond_counts = [
            [c for _, c in counter.most_common()] for counter in cond_counts]
    counts = [sum(c) for c in cond_counts]
    return entropy(counts) - conditional_entropy(cond_counts)


def l2(p, q):
    """
    Compute the L2 distance between two multinomial distributions specified as
    dictionaries of counts.
    """
    totalp = sum(p.values())
    totalq = sum(q.values())
    p_pnq = sum((v / totalp - q.get(k, 0.0) / totalq)**2 for k, v in p.items())
    q = sum((v / totalq)**2 for k, v in q.items() if k not in p)
    return math.sqrt(p_pnq + q)
