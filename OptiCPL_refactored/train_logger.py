"""
Training Logger Module for OptiCPL
Handles training log folder creation and management with timestamps
"""

import os
import datetime
from pathlib import Path
from typing import Optional


def create_train_log_dir(
    base_dir: Optional[str] = None,
    experiment_name: Optional[str] = None
) -> str:
    """
    Create a timestamped training log directory.

    Args:
        base_dir: Base directory for training logs (default: ~/Nano_HF/OptiCPL/Training_Logs)
        experiment_name: Optional experiment name to include in directory name

    Returns:
        Path to the created log directory
    """
    # Set default base directory
    if base_dir is None:
        base_dir = os.path.expanduser("~/Nano_HF/OptiCPL/Training_Logs")

    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    # Generate timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create directory name
    if experiment_name:
        dir_name = f"{timestamp}_{experiment_name}"
    else:
        dir_name = f"{timestamp}_training"

    # Full path
    log_dir = os.path.join(base_dir, dir_name)

    # Create subdirectories
    subdirs = [
        "models",
        "scalers",
        "results",
        "visualizations",
        "logs"
    ]

    for subdir in subdirs:
            subdir_path = os.path.join(log_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)

    print(f"Created training log directory: {log_dir}")
    return log_dir


def find_log_dir_by_experiment(experiment_name: str, base_dir: Optional[str] = None) -> Optional[str]:
    """
    Find the most recent training log directory for a given experiment name.

    Args:
        experiment_name: Experiment name to search for
        base_dir: Base directory for training logs (default: ~/Nano_HF/OptiCPL/Training_Logs)

    Returns:
        Path to the most recent matching log directory, or None if not found
    """
    if base_dir is None:
        base_dir = os.path.expanduser("~/Nano_HF/OptiCPL/Training_Logs")

    if not os.path.exists(base_dir):
        return None

    # Find all directories matching the experiment name
    matching_dirs = []
    for dirname in os.listdir(base_dir):
        if dirname.endswith(f"_{experiment_name}") and os.path.isdir(os.path.join(base_dir, dirname)):
            matching_dirs.append(dirname)

    if not matching_dirs:
        return None

    # Sort by timestamp (descending) and return the most recent
    matching_dirs.sort(reverse=True)
    return os.path.join(base_dir, matching_dirs[0])


def get_log_subdir(log_dir: str, subdir_type: str) -> str:
    """
    Get the path to a specific subdirectory within the log directory.

    Args:
        log_dir: Main log directory path
        subdir_type: Type of subdirectory (models, scalers, results, etc.)

    Returns:
        Path to the subdirectory
    """
    return os.path.join(log_dir, subdir_type)


def save_training_info(log_dir: str, config_dict: dict) -> None:
    """
    Save training configuration and information to a file.

    Args:
        log_dir: Log directory path
        config_dict: Configuration dictionary to save
    """
    info_file = os.path.join(log_dir, "training_info.txt")

    with open(info_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("OptiCPL Training Session Information\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Training Start Time: {datetime.datetime.now()}\n\n")

        f.write("Configuration:\n")
        f.write("-" * 30 + "\n")
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")

    print(f"Training info saved to: {info_file}")
