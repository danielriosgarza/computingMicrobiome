"""Compound opcode classifier model."""

from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from ..benchmarks.k_compound_opcode_bm import (
    train_compound_opcode_readout,
    run_episode_record_tagged,
)
from ..readouts.base import Readout


class KCompoundOpcode(BaseEstimator, ClassifierMixin):
    """Compound opcode classifier with two 4-bit opcodes and operands.

    Each opcode encodes the truth table for f(x, y) with ordering:
    (x, y) = 00, 01, 10, 11. The first opcode is applied to (a, b),
    and its output is then combined with c using the second opcode.

    Args:
        rule_number: ECA rule number (0-255).
        width: Number of cells in the automaton.
        boundary: Boundary condition ("periodic", "fixed_zero", "fixed_one",
            "mirror", or "random").
        recurrence: Number of input segments for injection.
        itr: Number of iterations between ticks.
        d_period: Delay between input and output window.
        repeats: Number of episode repeats per sample.
        feature_mode: Feature extraction mode ("cue_tick" or "window").
        output_window: Output window length when using windowed features.
        seed: RNG seed for dataset generation and sampling.
        readout_kind: "svm" or "evo".
        readout_config: Optional configuration for the readout.
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
        readout_kind: str = "svm",
        readout_config: dict | None = None,
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
        self.readout_kind = str(readout_kind)
        self.readout_config = readout_config

        self.reg_: Readout | None = None
        self.input_locations_: np.ndarray | None = None

    def fit(self, X=None, y=None):
        """Fit the classifier using the full 2048-case dataset.

        Args:
            X: Ignored (present for sklearn API compatibility).
            y: Ignored (present for sklearn API compatibility).

        Returns:
            KCompoundOpcode: Fitted estimator.
        """
        self.reg_, self.input_locations_ = train_compound_opcode_readout(
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
            readout_kind=self.readout_kind,
            readout_config=self.readout_config,
        )
        return self

    def predict(self, X):
        """Predict compound opcode outputs.

        Args:
            X: Array-like of shape (n_samples, 11) with
                [op1_3, op1_2, op1_1, op1_0, a, b, op2_3, op2_2, op2_1, op2_0, c].

        Returns:
            np.ndarray: Predicted outputs of shape (n_samples,).
        """
        if self.reg_ is None or self.input_locations_ is None:
            raise RuntimeError("Model not fitted: call fit() before predict().")

        X = np.asarray(X, dtype=np.int8)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != 11:
            raise ValueError(
                "Each sample must be [op1_3, op1_2, op1_1, op1_0, a, b, op2_3, op2_2, op2_1, op2_0, c] (length 11)."
            )

        rng = np.random.default_rng(self.seed)

        y_pred = np.zeros((X.shape[0],), dtype=np.int64)

        for i, row in enumerate(X):
            op1_bits = row[:4]
            a = int(row[4])
            b = int(row[5])
            op2_bits = row[6:10]
            c = int(row[10])

            ep = run_episode_record_tagged(
                op1_bits_msb_first=op1_bits,
                a=a,
                b=b,
                op2_bits_msb_first=op2_bits,
                c=c,
                rule_number=self.rule_number,
                width=self.width,
                boundary=self.boundary,
                itr=self.itr,
                d_period=self.d_period,
                rng=rng,
                input_locations=self.input_locations_,
                repeats=self.repeats,
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
