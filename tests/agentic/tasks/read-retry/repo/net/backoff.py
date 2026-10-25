BASE = 0.5
CAP = 12.0


def next_delay(attempt):
    """Exponential backoff with a cap."""
    return min(CAP, BASE * (2 ** attempt))


def jitter(x):
    return x * 0.9
