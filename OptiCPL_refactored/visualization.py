"""
Visualization module for Stage 1 Model
Handles plotting of training curves and prediction results
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    r2_scores: List[float],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
):
    """
    Plot training curves for loss and R² score.

    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        r2_scores: List of R² scores
        save_path: Path to save figure (optional)
        figsize: Figure size (width, height)
    """
    epochs = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Training and validation loss
    axes[0].plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # R² score
    axes[1].plot(epochs, r2_scores, 'g-', label='R² Score', linewidth=2)
    axes[1].set_title('R² Score on Test Set')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('R² Score')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Loss difference (val - train)
    loss_diff = np.array(val_losses) - np.array(train_losses)
    axes[2].plot(epochs, loss_diff, 'm-', label='Val - Train Loss', linewidth=2)
    axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[2].set_title('Validation-Training Loss Gap')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss Difference')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")

    plt.show()


def plot_prediction_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: List[str],
    save_path: Optional[str] = None,
    figsize_per_plot: Tuple[int, int] = (4, 4)
):
    """
    Plot predicted vs actual values for each target variable.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        target_columns: Names of target variables
        save_path: Path to save figure (optional)
        figsize_per_plot: Size of each subplot
    """
    n_targets = len(target_columns)
    n_cols = min(3, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols

    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_targets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (col, ax) in enumerate(zip(target_columns, axes)):
        # Scatter plot
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6, s=30)

        # Perfect prediction line
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # Metrics
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])

        ax.set_xlabel(f'True {col}')
        ax.set_ylabel(f'Predicted {col}')
        ax.set_title(f'{col}\nMSE: {mse:.4f}, R²: {r2:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_targets, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction comparison saved to {save_path}")

    plt.show()


def plot_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: List[str],
    save_path: Optional[str] = None,
    figsize_per_plot: Tuple[int, int] = (4, 4)
):
    """
    Plot residuals (errors) for each target variable.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        target_columns: Names of target variables
        save_path: Path to save figure (optional)
        figsize_per_plot: Size of each subplot
    """
    residuals = y_true - y_pred

    n_targets = len(target_columns)
    n_cols = min(3, n_targets)
    n_rows = (n_targets + n_cols - 1) // n_cols

    figsize = (figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    if n_targets == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (col, ax) in enumerate(zip(target_columns, axes)):
        # Histogram of residuals
        ax.hist(residuals[:, i], bins=30, alpha=0.7, edgecolor='black')

        # Add vertical line at 0
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)

        # Statistics
        mean_residual = np.mean(residuals[:, i])
        std_residual = np.std(residuals[:, i])

        ax.set_xlabel(f'Residuals for {col}')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{col}\nMean: {mean_residual:.4f}, Std: {std_residual:.4f}')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_targets, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual plots saved to {save_path}")

    plt.show()


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: List[str],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6)
):
    """
    Plot error distribution as a heatmap.

    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        target_columns: Names of target variables
        save_path: Path to save figure (optional)
        figsize: Figure size (width, height)
    """
    # Calculate errors (absolute percentage error)
    errors = np.abs((y_true - y_pred) / y_true) * 100

    # Calculate mean errors for each target
    mean_errors = np.mean(errors, axis=0)

    # Create DataFrame
    error_df = pd.DataFrame([mean_errors], columns=target_columns, index=['Mean Absolute % Error'])

    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(error_df, annot=True, fmt='.2f', cmap='YlOrRd', cbar=True)
    plt.title('Mean Absolute Percentage Error by Target Variable')
    plt.ylabel('')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution saved to {save_path}")

    plt.show()


def plot_learning_curve(
    train_sizes: np.ndarray,
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    metric: str = 'R² Score',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6)
):
    """
    Plot learning curve (model performance vs training set size).

    Args:
        train_sizes: Array of training set sizes
        train_scores: Training scores for each size
        val_scores: Validation scores for each size
        metric: Name of the metric being plotted
        save_path: Path to save figure (optional)
        figsize: Figure size (width, height)
    """
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=figsize)
    plt.plot(train_sizes, train_mean, 'o-', label=f'Training {metric}', linewidth=2)
    plt.plot(train_sizes, val_mean, 'o-', label=f'Validation {metric}', linewidth=2)

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)

    plt.xlabel('Training Set Size')
    plt.ylabel(metric)
    plt.title(f'Learning Curve: {metric}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning curve saved to {save_path}")

    plt.show()
