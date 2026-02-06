from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
import numpy as np

from ..eca import eca_rule_lkt, eca_step
from ..utils import create_input_locations, int_to_bits, flatten_history


class KXOR(BaseEstimator, ClassifierMixin):
    """k-bit parity (XOR) classifier driven through an ECA reservoir.

    Label definition (parity / XOR_k):
      y = 1  if an odd number of input bits are 1
          0  otherwise

    Notes:
    - `fit` ignores the provided X/y and trains on the full truth table (2**bits).
    - `predict(X)` expects `X` to be an array-like of bit-vectors of length `bits`.
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
        injection_interval: int = 0,
        injection_repetitions: int = 1,
        seed: int = 0,
    ):
        self.bits = int(bits)
        self.rule_number = int(rule_number)
        self.width = int(width)
        self.boundary = str(boundary)
        self.recurrence = int(recurrence)
        self.itr = int(itr)
        self.d_period = int(d_period)
        self.injection_interval = int(injection_interval)
        self.injection_repetitions = int(injection_repetitions)
        self.seed = int(seed)

        # learned / set during fit
        self.input_locations_: np.ndarray | None = None
        self._channel_idx_: np.ndarray | None = None
        self.reg_: SVC | None = None

    @staticmethod
    def _parity(bits_arr: np.ndarray) -> int:
        """Return XOR/parity of a bit-vector as 0/1."""
        return int(np.sum(bits_arr.astype(np.int8)) % 2)

    def _create_input_streams(self, bits_arr: np.ndarray) -> np.ndarray:
        # How long it takes to inject the k bits (with optional spacing)
        injection_duration = (self.bits - 1) * self.injection_interval + 1
        total_injection_duration = self.injection_repetitions * injection_duration

        # Add a 1-tick gap between repetitions (if any)
        if self.injection_repetitions > 1:
            total_injection_duration += self.injection_repetitions - 1

        # L ticks: injection + delay + cue
        L = total_injection_duration + self.d_period + 1  # +1 for the cue tick

        # channels: k data channels + 1 distractor + 1 cue
        streams = np.zeros((L, self.bits + 2), dtype=np.int8)

        tick = 0
        for _ in range(self.injection_repetitions):
            for i in range(self.bits):
                streams[tick, i] = int(bits_arr[i])
                if self.injection_interval > 0 and i < self.bits - 1:
                    tick += self.injection_interval

            if self.injection_repetitions > 1:
                tick += 1  # gap between repetitions

        # Distractor channel: on during delay only
        distractor_ch = self.bits
        streams[:, distractor_ch] = 1
        # Turn off during input injection
        streams[:total_injection_duration, distractor_ch] = 0

        # Cue channel: a single 1 at cue tick
        cue_ch = self.bits + 1
        cue_tick = total_injection_duration + self.d_period
        streams[cue_tick, cue_ch] = 1

        return streams

    def _run_episode(
        self,
        bits_arr: np.ndarray,
        rng: np.random.Generator,
        rule: np.ndarray,
    ) -> np.ndarray:
        if self.input_locations_ is None or self._channel_idx_ is None:
            raise RuntimeError("Model not fitted: input locations are missing. Call fit() first.")

        input_streams = self._create_input_streams(bits_arr)
        L, num_channels = input_streams.shape

        injection_duration = (self.bits - 1) * self.injection_interval + 1
        total_injection_duration = self.injection_repetitions * injection_duration
        if self.injection_repetitions > 1:
            total_injection_duration += self.injection_repetitions - 1
        cue_tick = total_injection_duration + self.d_period

        iter_between = self.itr + 1
        T = L * iter_between

        x = np.zeros(self.width, dtype=np.int8)

        # history contains the last `itr` states used by `flatten_history`
        history = [np.zeros(self.width, dtype=np.int8) for _ in range(self.itr)]
        output_features = None

        for t in range(T):
            if t % iter_between == 0:
                tick = t // iter_between
                if tick < L:
                    in_bits = input_streams[tick]  # shape: (num_channels,)
                    # XOR inject into reservoir at fixed locations
                    x[self.input_locations_] ^= in_bits[self._channel_idx_]

            history.append(x.copy())
            history = history[-self.itr :]

            if tick == cue_tick:
                output_features = flatten_history(history)

            x = eca_step(x, rule, self.boundary, rng=rng)

        return output_features

    def fit(self, X=None, y=None):
        rng = np.random.default_rng(self.seed)

        num_channels = self.bits + 2
        self.input_locations_ = create_input_locations(
            self.width, self.recurrence, num_channels, rng
        )
        # Map each injection site to a channel id (cyclic)
        self._channel_idx_ = np.arange(self.input_locations_.size) % num_channels

        rule = eca_rule_lkt(self.rule_number)

        X_train = []
        y_train = []

        for i in range(2**self.bits):
            bits_arr = int_to_bits(i, self.bits)
            X_train.append(self._run_episode(bits_arr, rng, rule))
            y_train.append(self._parity(bits_arr))

        self.reg_ = SVC(kernel="linear")
        self.reg_.fit(X_train, y_train)
        return self

    def predict(self, X):
        if self.reg_ is None:
            raise RuntimeError("Model not fitted: call fit() before predict().")

        rng = np.random.default_rng(self.seed)
        rule = eca_rule_lkt(self.rule_number)

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        y_pred = []
        for row in X:
            bits_arr = np.asarray(row, dtype=np.int8)
            if bits_arr.size != self.bits:
                raise ValueError(
                    f"Each input must have length {self.bits}, got {bits_arr.size}."
                )
            final_state_flat = self._run_episode(bits_arr, rng, rule)
            y_pred.append(int(self.reg_.predict([final_state_flat])[0]))

        return np.asarray(y_pred, dtype=np.int64)
