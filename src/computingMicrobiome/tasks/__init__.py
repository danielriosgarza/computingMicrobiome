"""Reusable task helpers built on top of core reservoir functionality."""

from .toy_addition import (
    addition_reservoir_features,
    bit_balance,
    bits_to_int,
    build_reservoir_dataset,
    constant_zero_baseline_accuracy,
    enumerate_addition_dataset,
    evaluate_linear_task,
    evaluate_models,
    full_adder_reservoir_features,
    int_to_bits,
    majority_baseline_accuracy,
    train_direct_linear_models,
    train_reservoir_linear_models,
)

__all__ = [
    "addition_reservoir_features",
    "bit_balance",
    "bits_to_int",
    "build_reservoir_dataset",
    "constant_zero_baseline_accuracy",
    "enumerate_addition_dataset",
    "evaluate_linear_task",
    "evaluate_models",
    "full_adder_reservoir_features",
    "int_to_bits",
    "majority_baseline_accuracy",
    "train_direct_linear_models",
    "train_reservoir_linear_models",
]
