def print_perf_counter_delta(delta: float) -> str:
    """Prints a performance counter delta in a human-readable format.

    Args:
        delta: delta in seconds

    Returns:
        Human-readable string
    """
    if delta < 1e-6:
        return f'{delta * 1e9:.2f} ns'
    elif delta < 1e-3:
        return f'{delta * 1e6:.2f} us'
    elif delta < 1:
        return f'{delta * 1e3:.2f} ms'
    else:
        return f'{delta:.2f} s'
