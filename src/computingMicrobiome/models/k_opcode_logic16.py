from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
import numpy as np

from ..benchmarks.k_opcode_logic16_bm import (
    train_programmed_logic_readout,
    run_episode_record_tagged,
)


class KOpcodeLogic16(BaseEstimator, ClassifierMixin):
    """Programmable logic gate with 4-bit opcode + 2 operand bits.

    Call pattern:
        alu.predict([[op3, op2, op1, op0, a, b], ...])

    Opcode bits are MSB-first (op3 op2 op1 op0).
    The opcode encodes the full truth table for f(a, b) with ordering:
        (a,b) = 00, 01, 10, 11
    op0 is the output for 00, op1 for 01, op2 for 10, op3 for 11.
    """

    def __init__(
        self,
        rule_number: int,
        width: int,
        boundary: str,
        recurrence: int,
        itr: int,
        d_period: int,
        repeats: int = 1,
        feature_mode: str = "cue_tick",
        output_window: int = 2,
        seed: int = 0,
    ):
        self.rule_number = int(rule_number)
        self.width = int(width)
        self.boundary = str(boundary)
        self.recurrence = int(recurrence)
        self.itr = int(itr)
        self.d_period = int(d_period)
        self.repeats = int(repeats)
        self.feature_mode = str(feature_mode)
        self.output_window = int(output_window)
        self.seed = int(seed)

        self.reg_: SVC | None = None
        self.input_locations_: np.ndarray | None = None

    def fit(self, X=None, y=None):
        # Train on full 64-case truth table.
        self.reg_, self.input_locations_ = train_programmed_logic_readout(
            rule_number=self.rule_number,
            width=self.width,
            boundary=self.boundary,
            recurrence=self.recurrence,
            itr=self.itr,
            d_period=self.d_period,
            repeats=self.repeats,
            feature_mode=self.feature_mode,
            output_window=self.output_window,
            seed_train=self.seed,
            order=None,
        )
        return self

    def predict(self, X):
        if self.reg_ is None or self.input_locations_ is None:
            raise RuntimeError("Model not fitted: call fit() before predict().")

        X = np.asarray(X, dtype=np.int8)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != 6:
            raise ValueError(
                "Each sample must be [op3, op2, op1, op0, a, b] (length 6)."
            )

        rng = np.random.default_rng(self.seed)

        y_pred = np.zeros((X.shape[0],), dtype=np.int64)

        for i, row in enumerate(X):
            op_bits = row[:4]
            a = int(row[4])
            b = int(row[5])

            ep = run_episode_record_tagged(
                op_bits_msb_first=op_bits,
                a=a,
                b=b,
                rule_number=self.rule_number,
                width=self.width,
                boundary=self.boundary,
                itr=self.itr,
                d_period=self.d_period,
                rng=rng,
                input_locations=self.input_locations_,
                repeats=self.repeats,
                order=None,
                reg=None,
                collect_states=False,
                x0_mode="zeros",
                feature_mode=self.feature_mode,
                output_window=self.output_window,
            )

            if self.feature_mode == "cue_tick":
                feat = ep["X_episode"].reshape(1, -1)
            else:
                feat = ep["X_episode"].reshape(1, -1)  # already window; flatten

            y_pred[i] = int(self.reg_.predict(feat)[0])

        return y_pred
