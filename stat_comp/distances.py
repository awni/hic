import numpy as np
import math


def multinomial_entropy(counts):
    counts = np.array(counts)
    probs = counts / np.sum(counts, keepdims=True)
    return -np.sum(probs * np.log2(probs))


def l2(p, q):
    """
    Compute the L2 distance between two multinomial
    distributions specified as dictionaries of counts.
    """
    totalp = sum(p.values())
    totalq = sum(q.values())
    p_pnq = sum((v / totalp - q.get(k, 0.0) / totalq)**2 for k, v in p.items())
    q = sum((v / totalq)**2 for k, v in q.items() if k not in p)
    return math.sqrt(p_pnq + q)
