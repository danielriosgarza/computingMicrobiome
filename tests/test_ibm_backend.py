from __future__ import annotations

import numpy as np

from computingMicrobiome.benchmarks.episode_runner import run_reservoir_episode
from computingMicrobiome.ibm.diffusion import diffuse_resources
from computingMicrobiome.ibm.dilution import apply_dilution
from computingMicrobiome.ibm.params import load_params
from computingMicrobiome.ibm.reproduction import apply_reproduction
from computingMicrobiome.ibm.state import GridState, make_zero_state
from computingMicrobiome.reservoirs.factory import make_reservoir
from computingMicrobiome.reservoirs.ibm_backend import IBMReservoirBackend
from computingMicrobiome.utils import create_input_locations


def _cfg(**overrides: object) -> dict:
    base = {
        "height": 4,
        "width_grid": 5,
        "n_species": 2,
        "n_resources": 2,
        "diff_numer": 1,
        "diff_denom": 5,
        "dilution_p": 0.2,
        "inject_scale": 4.0,
    }
    base.update(overrides)
    return base


def test_ibm_reproducible() -> None:
    cfg = _cfg()
    b1 = IBMReservoirBackend(config=cfg)
    b2 = IBMReservoirBackend(config=cfg)

    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    b1.reset(rng1, x0_mode="random")
    b2.reset(rng2, x0_mode="random")

    inj_rng = np.random.default_rng(123)
    locs = np.array([0, 3, 6, 9, 12, 15, 18], dtype=int)
    ch = np.arange(locs.size, dtype=int) % 4

    for _ in range(8):
        inputs = inj_rng.integers(0, 2, size=4, dtype=np.int8)
        b1.inject(inputs, locs, ch)
        b2.inject(inputs, locs, ch)
        b1.step(rng1)
        b2.step(rng2)

    np.testing.assert_array_equal(b1.get_state(), b2.get_state())


def test_ibm_bounds() -> None:
    backend = IBMReservoirBackend(config=_cfg(inject_scale=20.0, dilution_p=0.35))
    rng = np.random.default_rng(11)
    backend.reset(rng, x0_mode="random")

    locs = np.arange(backend.width, dtype=int)
    ch = np.arange(backend.width, dtype=int) % 4
    inputs = np.ones(4, dtype=np.int8)

    for _ in range(12):
        backend.inject(inputs, locs, ch)
        backend.step(rng)

    st = backend._state
    assert np.all((st.occ == -1) | ((st.occ >= 0) & (st.occ < backend.n_species)))
    assert int(st.E.min()) >= 0
    assert int(st.E.max()) <= backend.env.Emax
    assert int(st.R.min()) >= 0
    assert int(st.R.max()) <= backend.env.Rmax


def test_ibm_diffusion_conserves() -> None:
    env, _ = load_params(_cfg(diff_numer=1, diff_denom=2, dilution_p=0.0))
    state = make_zero_state(
        height=env.height,
        width_grid=env.width_grid,
        n_resources=env.n_resources,
    )
    rng = np.random.default_rng(5)
    state.R = rng.integers(
        0, 20, size=state.R.shape, dtype=np.uint16
    ).astype(np.uint8)

    total_before = int(state.R.astype(np.int64).sum())
    diffuse_resources(state, env)
    total_after = int(state.R.astype(np.int64).sum())

    assert total_after == total_before


def test_ibm_dilution_removes() -> None:
    env, _ = load_params(_cfg(dilution_p=1.0, feed_rate=0.0))
    state = make_zero_state(
        height=env.height,
        width_grid=env.width_grid,
        n_resources=env.n_resources,
    )
    state.R.fill(10)
    apply_dilution(state, env, np.random.default_rng(0))
    assert int(state.R.sum()) == 0


class _DummyRNG:
    def __init__(self, outputs: list[np.ndarray]):
        self._outputs = outputs

    def integers(
        self,
        low: int,
        high: int | None = None,
        size: int | tuple[int, ...] | None = None,
        dtype: type[np.integer] = np.int64,
    ) -> np.ndarray | int:
        out = np.asarray(self._outputs.pop(0), dtype=dtype)
        if size is None:
            return int(out.reshape(-1)[0])
        return out.reshape(size)


def test_ibm_birth_conflicts() -> None:
    env, species = load_params(
        {
            "height": 1,
            "width_grid": 3,
            "n_species": 2,
            "n_resources": 1,
            "div_threshold": [10, 10],
            "div_cost": [5, 5],
            "birth_energy": [7, 7],
        }
    )
    state = GridState(
        occ=np.array([[0, -1, 1]], dtype=np.int16),
        E=np.array([[50, 0, 40]], dtype=np.uint8),
        R=np.zeros((1, 1, 3), dtype=np.uint8),
    )

    rng = _DummyRNG(
        [
            np.array([1], dtype=np.int16),  # species 0 parent -> right (towards center)
            np.array([10], dtype=np.int64),  # species 0 random tie-break
            np.array([3], dtype=np.int16),  # species 1 parent -> left (towards center)
            np.array([65535], dtype=np.int64),  # species 1 random tie-break
        ]
    )
    apply_reproduction(state, species, env, rng)  # type: ignore[arg-type]

    assert int(state.occ[0, 1]) == 0
    assert int(state.E[0, 1]) == 7
    assert int(state.E[0, 0]) == 45
    assert int(state.E[0, 2]) == 40


def test_ibm_invasion_replaces_occupied_neighbor() -> None:
    env, species = load_params(
        {
            "height": 1,
            "width_grid": 2,
            "n_species": 2,
            "n_resources": 1,
            "allow_invasion": True,
            "invasion_energy_margin": 0,
            "div_threshold": [10, 255],  # species 1 cannot reproduce
            "div_cost": [0, 0],
            "birth_energy": [7, 7],
        }
    )
    state = GridState(
        occ=np.array([[0, 1]], dtype=np.int16),
        E=np.array([[50, 20]], dtype=np.uint8),
        R=np.zeros((1, 1, 2), dtype=np.uint8),
    )

    rng = _DummyRNG(
        [
            np.array([1], dtype=np.int16),  # species 0 parent -> right
            np.array([42], dtype=np.int64),  # random tie-break component
        ]
    )
    apply_reproduction(state, species, env, rng)  # type: ignore[arg-type]

    # Occupied target is replaced by offspring species 0.
    assert int(state.occ[0, 1]) == 0
    assert int(state.E[0, 1]) == 7


def test_ibm_protocol_alignment() -> None:
    cfg = {"height": 3, "width_grid": 4, "n_species": 2, "n_resources": 2}
    reservoir = make_reservoir(
        reservoir_kind="ibm",
        rule_number=110,
        width=999,
        boundary="periodic",
        reservoir_config=cfg,
    )
    reservoir.reset(np.random.default_rng(0), x0_mode="zeros")
    x = reservoir.get_state()

    assert reservoir.width == 12
    assert x.shape == (12,)
    assert x.dtype == np.float32


def test_ibm_backend_integration() -> None:
    cfg = {"height": 4, "width_grid": 4, "n_species": 2, "n_resources": 2}
    width = cfg["height"] * cfg["width_grid"]
    reservoir = make_reservoir(
        reservoir_kind="ibm",
        rule_number=110,
        width=width,
        boundary="periodic",
        reservoir_config=cfg,
    )

    rng = np.random.default_rng(42)
    input_streams = rng.integers(0, 2, size=(6, 4), dtype=np.int8)
    input_locations = create_input_locations(width, recurrence=2, input_channels=4, rng=rng)

    ep = run_reservoir_episode(
        input_streams=input_streams,
        reservoir=reservoir,
        itr=3,
        input_locations=input_locations,
        rng=rng,
        reg=None,
        collect_states=True,
        x0_mode="zeros",
    )

    assert ep["X_tick"].shape == (6, 3 * width)
    assert reservoir.get_state().shape == (width,)


def test_ibm_input_trace_delay_line() -> None:
    cfg = {
        "height": 2,
        "width_grid": 2,
        "n_species": 1,
        "n_resources": 1,
        "state_width_mode": "raw",
        "input_trace_depth": 3,
        "input_trace_channels": 2,
        "input_trace_decay": 1.0,
        "inject_scale": 0.0,
        "dilution_p": 0.0,
        "diff_numer": 0,
        "maint_cost": [0],
        "uptake_rate": [0],
        "yield_energy": [0],
        "div_threshold": [255],
        "div_cost": [0],
        "birth_energy": [0],
    }
    backend = IBMReservoirBackend(config=cfg)
    rng = np.random.default_rng(0)
    backend.reset(rng, x0_mode="zeros")

    # raw width = occ_onehot(4) + E(4) + R(4) + counts(1) = 13
    # plus trace width = 3 * 2 = 6
    assert backend.width == 19

    backend.inject(
        input_values=np.array([1, 0], dtype=np.int8),
        input_locations=np.array([0], dtype=int),
        channel_idx=np.array([0], dtype=int),
    )
    st0 = backend.get_state()
    np.testing.assert_array_equal(st0[-6:], np.array([1, 0, 0, 0, 0, 0], dtype=np.float32))

    backend.step(rng)
    st1 = backend.get_state()
    np.testing.assert_array_equal(st1[-6:], np.array([0, 0, 1, 0, 0, 0], dtype=np.float32))
