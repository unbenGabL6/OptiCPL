"""
Inference module for Stage 1 Model
Handles loading trained models and making predictions
"""

from typing import Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from .model import NeuralNet
from .data_utils import load_inference_data, prepare_inference_features, load_scalers
from .trainer import load_model_for_inference


def make_predictions(
    model: NeuralNet,
    inputs: torch.Tensor,
    scaler_y,
    device: torch.device,
    batch_size: int = 32
) -> np.ndarray:
    """
    Make predictions on input data.

    Args:
        model: Trained neural network model
        inputs: Input feature tensor
        scaler_y: Label scaler for inverse transformation
        device: Device to run inference on
        batch_size: Batch size for inference

    Returns:
        Predicted values in original scale
    """
    model.eval()

    # Create dataset and loader
    dataset = TensorDataset(inputs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []

    with torch.no_grad():
        for batch in loader:
            X_batch = batch[0].to(device)
            outputs = model(X_batch)
            all_predictions.append(outputs.cpu())

    # Concatenate all predictions
    predictions = torch.cat(all_predictions, dim=0)

    # Inverse transform to original scale
    predictions_actual = scaler_y.inverse_transform(predictions.numpy())

    return predictions_actual


def run_inference_from_dataframe(
    df: pd.DataFrame,
    model_path: str,
    input_size: int,
    output_size: int,
    config,
    device: torch.device
) -> np.ndarray:
    """
    Run inference on a DataFrame and return predictions.

    Args:
        df: DataFrame containing inference data
        model_path: Path to trained model
        input_size: Number of input features
        output_size: Number of output targets
        config: Configuration object
        device: Device to run inference on

    Returns:
        Predicted values
    """
    # Load model and scalers
    model = load_model_for_inference(
        model_path, input_size, output_size, config, device
    )
    scaler_x, scaler_y = load_scalers(config.scaler_x_path, config.scaler_y_path)

    # Prepare features
    inputs = prepare_inference_features(
        df, scaler_x, config.img_column, "IMG_seed28",
        config.glum_column, config.optical_column
    ).to(device)

    # Make predictions
    predictions = make_predictions(model, inputs, scaler_y, device, config.batch_size)

    return predictions


def run_inference_from_file(
    data_path: str,
    model_path: str,
    input_size: int,
    output_size: int,
    config,
    device: torch.device
) -> np.ndarray:
    """
    Run inference on data from file and return predictions.

    Args:
        data_path: Path to inference data file
        model_path: Path to trained model
        input_size: Number of input features
        output_size: Number of output targets
        config: Configuration object
        device: Device to run inference on

    Returns:
        Predicted values
    """
    # Load data
    df = load_inference_data(data_path)

    # Run inference
    predictions = run_inference_from_dataframe(
        df, model_path, input_size, output_size, config, device
    )

    return predictions


def save_predictions(
    predictions: np.ndarray,
    output_path: str,
    target_columns: Optional[list] = None,
    add_index: bool = False
):
    """
    Save predictions to CSV file.

    Args:
        predictions: Predicted values array
        output_path: Path to save predictions
        target_columns: Column names for the predictions
        add_index: Whether to include index column in CSV
    """
    # Create DataFrame
    if target_columns is not None:
        df_predictions = pd.DataFrame(predictions, columns=target_columns)
    else:
        df_predictions = pd.DataFrame(predictions)

    # Save to CSV
    df_predictions.to_csv(output_path, index=add_index)
    print(f"Predictions saved to {output_path}")


def batch_predict(
    model: NeuralNet,
    dataloader: DataLoader,
    scaler_y,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions on a dataset and return both scaled and unscaled predictions.

    Args:
        model: Trained neural network model
        dataloader: DataLoader for the dataset
        scaler_y: Label scaler for inverse transformation
        device: Device to run inference on

    Returns:
        Tuple of (scaled_predictions, unscaled_predictions)
    """
    model.eval()
    all_scaled_predictions = []

    with torch.no_grad():
        for X_batch, _ in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_scaled_predictions.append(outputs.cpu())

    # Concatenate predictions
    scaled_predictions = torch.cat(all_scaled_predictions, dim=0).numpy()

    # Inverse transform to original scale
    unscaled_predictions = scaler_y.inverse_transform(scaled_predictions)

    return scaled_predictions, unscaled_predictions


def predict_with_uncertainty(
    model: NeuralNet,
    inputs: torch.Tensor,
    scaler_y,
    device: torch.device,
    num_samples: int = 10,
    batch_size: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make predictions with uncertainty estimation using dropout at inference time.

    Args:
        model: Trained neural network model (should have dropout layers)
        inputs: Input feature tensor
        scaler_y: Label scaler for inverse transformation
        device: Device to run inference on
        num_samples: Number of stochastic forward passes
        batch_size: Batch size for inference

    Returns:
        Tuple of (mean_predictions, std_predictions) in original scale
    """
    model.train()  # Keep dropout active

    # Create dataset and loader
    dataset = TensorDataset(inputs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_predictions = []

    # Multiple stochastic forward passes
    for _ in range(num_samples):
        sample_predictions = []

        with torch.no_grad():
            for batch in loader:
                X_batch = batch[0].to(device)
                outputs = model(X_batch)
                sample_predictions.append(outputs.cpu())

        # Concatenate and inverse transform
        sample_preds = torch.cat(sample_predictions, dim=0).numpy()
        sample_preds_actual = scaler_y.inverse_transform(sample_preds)
        all_predictions.append(sample_preds_actual)

    # Convert to array and calculate statistics
    predictions_array = np.array(all_predictions)  # shape: (num_samples, num_examples, num_targets)
    mean_predictions = np.mean(predictions_array, axis=0)
    std_predictions = np.std(predictions_array, axis=0)

    model.eval()  # Set back to eval mode

    return mean_predictions, std_predictions
