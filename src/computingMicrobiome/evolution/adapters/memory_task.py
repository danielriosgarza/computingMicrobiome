"""Task sampler for the k-bit memory benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..base import TaskSamplerProtocol
from ...utils import int_to_bits


@dataclass
class MemoryTaskSampler(TaskSamplerProtocol):
    """Sample bit-vector memory instances for support and challenge sets.

    Each sampled instance corresponds to a full memory episode input
    represented as a bit vector of shape ``(bits,)``. Representation
    adapters are responsible for expanding these into per-bit feature
    rows and labels.
    """

    bits: int
    seed: int = 0

    def _sample_bits(
        self, rng: np.random.Generator, n_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.bits < 1:
            raise ValueError("bits must be >= 1")
        vals = rng.integers(0, 2**self.bits, size=n_samples, dtype=np.int64)
        raw_x = np.stack([int_to_bits(int(v), self.bits) for v in vals], axis=0)
        # raw_y is currently unused for the memory benchmark; kept for API parity.
        raw_y = vals.astype(np.int64)
        return raw_x, raw_y

    def sample_support(
        self, rng: np.random.Generator, n_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._sample_bits(rng, n_samples)

    def sample_challenge(
        self, rng: np.random.Generator, n_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._sample_bits(rng, n_samples)


__all__ = ["MemoryTaskSampler"]

