import numpy as np


def eca_rule_lkt(rule_number: int) -> np.ndarray:
    """
    Generate a lookup table for any elementary cellular automata.
    (reversed 8-bit number)
    """
    if not (0 <= rule_number <= 255):
        raise ValueError("rule_number must be between 0 and 255")
    bits = [(rule_number >> i) & 1 for i in range(8)]
    return np.array(bits, dtype=np.int8)


def eca_step(x: np.ndarray, rule: np.ndarray, boundary: str, rng=None) -> np.ndarray:
    """
    Single step of an elementary cellular automata.
    x         : input state (binary vector)
    rule      : lookup table (output of eca_rule_lkt)
    boundary  : boundary condition to use from periodic,
                fixed_zero, fixed_one, mirror, random.
    rng       : random number generator, used for random boundary.
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
    """
    Simulate an elementary cellular automata.
    x0       : initial state
    rule     : rule lookup table
    T        : "time" (update steps)
    boundary : boudary condition to use from periodic,
               fixed_zero, fixed_one, mirror, random.
    rng      : random number generator, used for random boundary.
    """
    x = x0.astype(np.int8).copy()
    states = np.zeros((T + 1, x.size), dtype=np.int8)
    states[0] = x
    for _ in range(1, T + 1):
        x = eca_step(x, rule, boundary, rng)
        states[_] = x
    return states
