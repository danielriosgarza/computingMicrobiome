"""Shared reservoir episode runner."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..readouts.base import Readout
from ..reservoirs.base import ReservoirBackend
from ..utils import flatten_history


def run_reservoir_episode(
    *,
    input_streams: np.ndarray,
    reservoir: ReservoirBackend,
    itr: int,
    input_locations: np.ndarray,
    rng: np.random.Generator,
    reg: Optional[Readout] = None,
    collect_states: bool = True,
    x0_mode: str = "zeros",
) -> dict:
    """Run one reservoir episode with unified tick/inject cadence."""
    L = int(input_streams.shape[0])
    iter_between = int(itr) + 1
    T = L * iter_between

    reservoir.reset(rng, x0_mode=x0_mode)
    state0 = reservoir.get_state()
    width = int(reservoir.width)
    if state0.shape != (width,):
        raise ValueError("reservoir.get_state() must return shape (width,)")

    state_dtype = state0.dtype
    history = [np.zeros(width, dtype=state_dtype) for _ in range(itr)]

    inputs_tick = np.zeros_like(input_streams)
    X_tick = np.zeros((L, itr * width), dtype=state_dtype)
    y_pred = np.full(L, -1, dtype=np.int8)
    states = np.zeros((T, width), dtype=state_dtype) if collect_states else None

    channel_idx = np.arange(input_locations.size) % input_streams.shape[1]
    tick = 0

    for i in range(T):
        if i % iter_between == 0:
            in_bits = input_streams[tick]
            inputs_tick[tick] = in_bits
            reservoir.inject(in_bits, input_locations, channel_idx)

        x = reservoir.get_state()
        history.append(x.copy())
        history = history[-itr:]

        if i % iter_between == 0:
            feat = flatten_history(history)
            X_tick[tick] = feat
            if reg is not None:
                y_pred[tick] = int(reg.predict([feat])[0])
            tick += 1

        if collect_states:
            states[i] = x

        reservoir.step(rng)

    return {
        "inputs_tick": inputs_tick,
        "X_tick": X_tick,
        "y_pred_tick": y_pred,
        "states": states,
        "L": L,
        "T": T,
        "iter_between": iter_between,
    }
