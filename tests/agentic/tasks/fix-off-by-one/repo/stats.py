def moving_average(xs, n):
    """Mean of every window of n consecutive values."""
    out = []
    for i in range(len(xs) - n):
        out.append(sum(xs[i:i + n]) / n)
    return out


def clamp(x, lo, hi):
    return max(lo, min(hi, x))
