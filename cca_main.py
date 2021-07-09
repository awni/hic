"""
Measure the hierarchical information content and statistical complexity of
cyclic cellular automata (CCA). For the CCA, we use the same settings as [1].

[1] Shalizi, C.R., Shalizi, K.L. and Haslinger, R., 2004. Quantifying
    self-organization with optimal predictors. Physical review letters, 93(11),
    p.118701.
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the statistical complexity computation")
    parser.add_argument("--seed",  type=int, default=2021,
        help="Manual seed for RNGs")
    args = parser.parse_args()

    n = 100 # 2D Grid size
    k = 4   # Number of colors
    T = 2   # Update threshold
    dp = 2  # Past light-cone depth (includes current state)
    df = 1  # Future light-cone depth

    random.seed(args.seed)
    np.random.seed(args.seed + 1)
    state = np.random.randint(low=0, high=k, size=(n, n))
    cca = models.CCA(k, T)
    states = [state]
    for _ in range(10):
        states.append(cca.update(states[-1]))

    for i in range(1, len(states) - df + 1):
        past_states = states[max(i - dp, 0):i]  # includes current state
        future_states = states[i:i+df]
        sc = compute_local_statistical_complexity(past_states, future_states)
        print(f"Iterations {i}: Statistical Complexity {sc}")
