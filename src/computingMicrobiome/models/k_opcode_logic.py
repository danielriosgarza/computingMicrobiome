"""Programmable 3-bit opcode logic classifier."""

from __future__ import annotations

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from ..benchmarks.k_opcode_logic_bm import (
    train_programmed_logic_readout,
    run_episode_record_tagged,
)
from ..readouts.base import Readout


class KOpcodeLogic(BaseEstimator, ClassifierMixin):
    """Programmable logic gate with a 3-bit opcode and 2 operand bits.

    Opcode bits are MSB-first and select one of 8 operations:

    - 000 AND
    - 001 OR
    - 010 XOR
    - 011 NAND
    - 100 NOR
    - 101 XNOR
    - 110 A
    - 111 B

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

    Example:
        >>> alu = KOpcodeLogic(110, 256, "periodic", 8, 8, 20).fit()
        >>> alu.predict([[0, 0, 0, 1, 1]]).tolist()
        [1]
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
        """Fit the classifier using the full 32-case truth table.

        Args:
            X: Ignored (present for sklearn API compatibility).
            y: Ignored (present for sklearn API compatibility).

        Returns:
            KOpcodeLogic: Fitted estimator.
        """
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
            readout_kind=self.readout_kind,
            readout_config=self.readout_config,
        )
        return self

    def predict(self, X):
        """Predict logic outputs for opcode/operand inputs.

        Args:
            X: Array-like of shape (n_samples, 5) with
                [op2, op1, op0, a, b] per row.

        Returns:
            np.ndarray: Predicted outputs of shape (n_samples,).
        """
        if self.reg_ is None or self.input_locations_ is None:
            raise RuntimeError("Model not fitted: call fit() before predict().")

        X = np.asarray(X, dtype=np.int8)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != 5:
            raise ValueError("Each sample must be [op2, op1, op0, a, b] (length 5).")

        rng = np.random.default_rng(self.seed)

        y_pred = np.zeros((X.shape[0],), dtype=np.int64)

        for i, row in enumerate(X):
            op_bits = row[:3]
            a = int(row[3])
            b = int(row[4])

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
