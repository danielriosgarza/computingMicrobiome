from .k_bit_memory_bm import (
    build_dataset_output_window_only,
    evaluate_memory_trials,
    run_episode_record,
    train_memory_readout,
)
from .k_compound_opcode_bm import (
    apply_compound_opcode,
    apply_opcode,
    build_dataset_compound_opcode,
    run_episode_record_tagged,
    train_compound_opcode_readout,
)
from .k_opcode_logic_bm import (
    apply_opcode as apply_opcode_8,
    build_dataset_programmed_logic,
    run_episode_record_tagged as run_episode_record_tagged_8,
    train_programmed_logic_readout,
)
from .k_opcode_logic16_bm import (
    apply_opcode as apply_opcode_16,
    build_dataset_programmed_logic as build_dataset_programmed_logic_16,
    run_episode_record_tagged as run_episode_record_tagged_16,
    train_programmed_logic_readout as train_programmed_logic_readout_16,
)
from .k_serial_adder_bm import (
    build_dataset_serial_adder,
    run_episode_record_serial_adder,
)

__all__ = [
    "apply_compound_opcode",
    "apply_opcode",
    "apply_opcode_8",
    "apply_opcode_16",
    "build_dataset_compound_opcode",
    "build_dataset_programmed_logic",
    "build_dataset_programmed_logic_16",
    "build_dataset_output_window_only",
    "build_dataset_serial_adder",
    "evaluate_memory_trials",
    "run_episode_record",
    "run_episode_record_tagged",
    "run_episode_record_tagged_8",
    "run_episode_record_tagged_16",
    "run_episode_record_serial_adder",
    "train_compound_opcode_readout",
    "train_memory_readout",
    "train_programmed_logic_readout",
    "train_programmed_logic_readout_16",
]
