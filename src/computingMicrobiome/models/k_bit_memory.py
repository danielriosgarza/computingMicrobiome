"""K-bit memory classifier model."""

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from ..benchmarks.k_bit_memory_bm import (
    train_memory_readout,
    run_episode_record,
)
from ..readouts.base import Readout


class KBitMemory(BaseEstimator, ClassifierMixin):
    """Memory-only task for k-bit recall using an ECA reservoir.

    Args:
        bits: Number of bits to store and recall.
        rule_number: ECA rule number (0-255).
        width: Number of cells in the automaton.
        boundary: Boundary condition ("periodic", "fixed_zero", "fixed_one",
            "mirror", or "random").
        recurrence: Number of input segments for injection.
        itr: Number of iterations between ticks.
        d_period: Delay between input and recall window.
        seed: RNG seed for dataset generation and sampling.
        readout_kind: "svm" or "evo".
        readout_config: Optional configuration for the readout.
    """

    def __init__(
        self,
        bits: int,
        rule_number: int,
        width: int,
        boundary: str,
        recurrence: int,
        itr: int,
        d_period: int,
        seed: int = 0,
        readout_kind: str = "svm",
        readout_config: dict | None = None,
        reservoir_kind: str = "eca",
        reservoir_config: dict | None = None,
    ):
        self.bits = bits
        self.rule_number = rule_number
        self.width = width
        self.boundary = boundary
        self.recurrence = recurrence
        self.itr = itr
        self.d_period = d_period
        self.seed = seed
        self.readout_kind = str(readout_kind)
        self.readout_config = readout_config
        self.reservoir_kind = str(reservoir_kind)
        self.reservoir_config = reservoir_config

        self.reg_: Readout | None = None
        self.input_locations_: np.ndarray | None = None

    def fit(self, X=None, y=None):
        """Fit the linear readout using the memory benchmark dataset.

        Args:
            X: Ignored (present for sklearn API compatibility).
            y: Ignored (present for sklearn API compatibility).

        Returns:
            KBitMemory: Fitted estimator.
        """
        self.reg_, self.input_locations_ = train_memory_readout(
            self.bits,
            self.rule_number,
            self.width,
            self.boundary,
            self.recurrence,
            self.itr,
            self.d_period,
            seed_train=self.seed,
            readout_kind=self.readout_kind,
            readout_config=self.readout_config,
            reservoir_kind=self.reservoir_kind,
            reservoir_config=self.reservoir_config,
        )
        return self

    def predict(self, X):
        """Predict recalled bits for a batch of inputs.

        Args:
            X: Array-like of shape (n_samples, n_bits) with input bit vectors.

        Returns:
            np.ndarray: Predicted bit vectors of shape (n_samples, bits).
        """
        y_pred = []
        if self.reg_ is None or self.input_locations_ is None:
            raise RuntimeError("Model not fitted: call fit() before predict().")

        rng = np.random.default_rng(self.seed)

        for bits_arr in X:
            ep = run_episode_record(
                bits_arr=bits_arr,
                rule_number=self.rule_number,
                width=self.width,
                boundary=self.boundary,
                itr=self.itr,
                d_period=self.d_period,
                rng=rng,
                input_locations=self.input_locations_,
                reg=self.reg_,
                collect_states=False,
                x0_mode="zeros",
                reservoir_kind=self.reservoir_kind,
                reservoir_config=self.reservoir_config,
            )

            X_out = ep["X_tick"][-self.bits:]  # features in output window
            pred_bits = self.reg_.predict(X_out).astype(np.int8)
            y_pred.append(pred_bits)

        return np.array(y_pred)
