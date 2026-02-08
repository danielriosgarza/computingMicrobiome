"""Elementary cellular automata utilities."""

import numpy as np


def eca_rule_lkt(rule_number: int) -> np.ndarray:
    """Generate an ECA rule lookup table.

    Args:
        rule_number: Integer in [0, 255] specifying the rule.

    Returns:
        np.ndarray: Lookup table of shape (8,) with dtype int8.
    """
    if not (0 <= rule_number <= 255):
        raise ValueError("rule_number must be between 0 and 255")
    bits = [(rule_number >> i) & 1 for i in range(8)]
    return np.array(bits, dtype=np.int8)


def eca_step(x: np.ndarray, rule: np.ndarray, boundary: str, rng=None) -> np.ndarray:
    """Advance one step of an elementary cellular automaton.

    Args:
        x: Current binary state of shape (width,).
        rule: Lookup table from `eca_rule_lkt`.
        boundary: Boundary condition ("periodic", "fixed_zero", "fixed_one",
            "mirror", or "random").
        rng: Optional random generator for "random" boundary.

    Returns:
        np.ndarray: Next state of shape (width,) with dtype int8.
    """
    if boundary == "periodic":
        left = np.roll(x, 1)
        right = np.roll(x, -1)
    elif boundary == "fixed_zero":
        left = np.concatenate(([0], x[:-1]))
        right = np.concatenate((x[1:], [0]))
    elif boundary == "fixed_one":
        left = np.concatenate(([1], x[:-1]))
        right = np.concatenate((x[1:], [1]))
    elif boundary == "mirror":
        left = np.concatenate([[x[1]], x[:-1]])
        right = np.concatenate([x[1:], [x[-2]]])
    elif boundary == "random":
        if rng is None:
            rng = np.random.default_rng()
        left_boundary = rng.integers(0, 2, dtype=np.int8)
        right_boundary = rng.integers(0, 2, dtype=np.int8)
        left = np.concatenate([[left_boundary], x[:-1]])
        right = np.concatenate([x[1:], [right_boundary]])
    else:
        raise ValueError(f"Unknown boundary condition: {boundary}")

    idx = (left << 2) | (x << 1) | right
    return rule[idx]


def eca_run(
    x0: np.ndarray, rule: np.ndarray, T: int, boundary: str, rng=None
) -> np.ndarray:
    """Simulate an elementary cellular automaton for T steps.

    Args:
        x0: Initial binary state of shape (width,).
        rule: Lookup table from `eca_rule_lkt`.
        T: Number of update steps.
        boundary: Boundary condition ("periodic", "fixed_zero", "fixed_one",
            "mirror", or "random").
        rng: Optional random generator for "random" boundary.

    Returns:
        np.ndarray: State history of shape (T + 1, width).
    """
    x = x0.astype(np.int8).copy()
    states = np.zeros((T + 1, x.size), dtype=np.int8)
    states[0] = x
    for _ in range(1, T + 1):
        x = eca_step(x, rule, boundary, rng)
        states[_] = x
    return states
