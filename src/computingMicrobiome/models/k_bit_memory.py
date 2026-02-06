from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

from ..benchmarks.k_bit_memory_bm import (
    train_memory_readout,
    run_episode_record,
)


class KBitMemory(BaseEstimator, ClassifierMixin):
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
    ):
        self.bits = bits
        self.rule_number = rule_number
        self.width = width
        self.boundary = boundary
        self.recurrence = recurrence
        self.itr = itr
        self.d_period = d_period
        self.seed = seed

    def fit(self, X=None, y=None):
        """
        Train the KBitMemory model.
        This will train a linear readout on the output of the ECA.
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
        )
        return self

    def predict(self, X):
        """
        Predict the output for a given set of k-bit strings.
        X should be a 2D numpy array of shape (n_samples, n_bits).
        """
        y_pred = []
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
            )

            X_out = ep["X_tick"][-self.bits:]  # features in output window
            pred_bits = self.reg_.predict(X_out).astype(np.int8)
            y_pred.append(pred_bits)

        return np.array(y_pred)
