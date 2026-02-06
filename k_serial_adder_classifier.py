from computingMicrobiome.models.k_serial_adder import KSerialAdder

__all__ = ["KSerialAdder"]
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

from k_serial_adder_bm import (
    build_dataset_serial_adder,
    run_episode_record_serial_adder,
)


class KSerialAdder(BaseEstimator, ClassifierMixin):
    """Serial adder classifier that predicts each output bit at cue ticks.

    Call pattern:
        model.predict([a0, a1, ...], [b0, b1, ...])
        model.predict([("0110", "0011"), ("0001", "0001")])
    """

    def __init__(
        self,
        rule_number: int,
        width: int,
        boundary: str,
        recurrence: int,
        itr: int,
        d_period: int,
        bits: int = 8,
        n_train: int = 500,
        seed: int = 0,
    ):
        self.rule_number = int(rule_number)
        self.width = int(width)
        self.boundary = str(boundary)
        self.recurrence = int(recurrence)
        self.itr = int(itr)
        self.d_period = int(d_period)
        self.bits = int(bits)
        self.n_train = int(n_train)
        self.seed = int(seed)

        self.reg_: SVC | None = None
        self.input_locations_: np.ndarray | None = None

    def fit(self, X=None, y=None):
        X_train, y_train, input_locations = build_dataset_serial_adder(
            bits=self.bits,
            n_samples=self.n_train,
            rule_number=self.rule_number,
            width=self.width,
            boundary=self.boundary,
            recurrence=self.recurrence,
            itr=self.itr,
            d_period=self.d_period,
            seed=self.seed,
        )
        self.reg_ = SVC(kernel="linear")
        self.reg_.fit(X_train, y_train)
        self.input_locations_ = input_locations
        return self

    def _coerce_pair_lists(self, A, B=None) -> Tuple[Sequence, Sequence]:
        if B is not None:
            return A, B

        arr = np.asarray(A, dtype=object)
        if arr.ndim == 2 and arr.shape[1] == 2:
            return arr[:, 0].tolist(), arr[:, 1].tolist()

        raise ValueError("Provide A and B lists, or X as shape (n, 2).")

    def _parse_value(self, v) -> int:
        if isinstance(v, str):
            if len(v) != self.bits:
                raise ValueError(f"Bitstring must be length {self.bits}.")
            return int(v, 2)
        iv = int(v)
        if iv < 0 or iv >= 2**self.bits:
            raise ValueError(f"Value {iv} out of range for {self.bits} bits.")
        return iv

    def predict(self, A, B=None):
        if self.reg_ is None or self.input_locations_ is None:
            raise RuntimeError("Model not fitted: call fit() before predict().")

        A_list, B_list = self._coerce_pair_lists(A, B)
        if len(A_list) != len(B_list):
            raise ValueError("A and B must have the same length.")

        rng = np.random.default_rng(self.seed)

        preds = np.zeros((len(A_list), self.bits), dtype=np.int8)

        for i, (a_val, b_val) in enumerate(zip(A_list, B_list)):
            a = self._parse_value(a_val)
            b = self._parse_value(b_val)

            ep = run_episode_record_serial_adder(
                a=a,
                b=b,
                bits=self.bits,
                rule_number=self.rule_number,
                width=self.width,
                boundary=self.boundary,
                itr=self.itr,
                d_period=self.d_period,
                rng=rng,
                input_locations=self.input_locations_,
                reg=None,
                collect_states=False,
                x0_mode="zeros",
            )

            cue_idx = ep["cue_indices"]
            X_out = ep["X_tick"][cue_idx]  # shape (bits, itr*width)
            preds[i] = self.reg_.predict(X_out).astype(np.int8)

        return preds
