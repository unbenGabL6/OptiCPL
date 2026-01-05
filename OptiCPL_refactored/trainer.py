"""
Training and evaluation module for Stage 1 Model
Handles model training, validation, evaluation, and checkpointing
"""

import os
import pickle
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from .model import NeuralNet
from .data_utils import save_scalers


def initialize_training(
    model: NeuralNet,
    learning_rate: float,
    device: torch.device
) -> Tuple[nn.Module, optim.Optimizer, nn.Module]:
    """
    Initialize model, optimizer, and loss function for training.

    Args:
        model: Neural network model
        learning_rate: Learning rate for optimizer
        device: Device to train on

    Returns:
        Tuple of (model_on_device, optimizer, criterion)
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    return model, optimizer, criterion


def train_epoch(
    model: NeuralNet,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train model for one epoch.

    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on

    Returns:
        Average training loss for the epoch
    """
    model.train()
    epoch_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(train_loader)


def validate_model(
    model: NeuralNet,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> float:
    """
    Validate model on validation set.

    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

    return val_loss / len(val_loader)


def evaluate_model(
    model: NeuralNet,
    test_loader: DataLoader,
    scaler_y,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float]:
    """
    Evaluate model on test set and compute metrics.

    Args:
        model: Neural network model
        test_loader: Test data loader
        scaler_y: Label scaler for inverse transformation
        criterion: Loss function
        device: Device to evaluate on

    Returns:
        Tuple of (test_loss, mse, r2_score)
    """
    model.eval()
    test_loss = 0.0

    with torch.no_grad():
        X_test = []
        y_test = []
        y_pred = []

        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)

            # Collect data for later inverse transformation
            X_test.append(X_batch.cpu())
            y_test.append(y_batch.cpu())
            y_pred.append(outputs.cpu())

            loss = criterion(outputs, y_batch)
            test_loss += loss.item()

        # Concatenate all batches
        X_test = torch.cat(X_test, dim=0)
        y_test = torch.cat(y_test, dim=0)
        y_pred = torch.cat(y_pred, dim=0)

        # Inverse transform to original scale
        y_test_actual = scaler_y.inverse_transform(y_test.numpy())
        y_pred_actual = scaler_y.inverse_transform(y_pred.numpy())

        # Calculate metrics
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        r2 = r2_score(y_test_actual, y_pred_actual)

    return test_loss / len(test_loader), mse, r2


def train_model(
    model: NeuralNet,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    config,
    device: torch.device,
    save_model: bool = True
) -> Tuple[NeuralNet, List[float], List[float], List[float], List[float]]:
    """
    Train model with early stopping and tracking.

    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        config: Configuration object
        device: Device to train on
        save_model: Whether to save the model

    Returns:
        Tuple of (trained_model, train_losses, val_losses, mse_scores, r2_scores)
    """
    # Initialize training
    model, optimizer, criterion = initialize_training(model, config.learning_rate, device)

    # Initialize tracking variables
    train_losses = []
    val_losses = []
    mse_scores = []
    r2_scores = []

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0

    # Create output directory if it doesn't exist
    os.makedirs(config.model_save_dir, exist_ok=True)

    print(f"Starting training for {config.num_epochs} epochs...")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(config.num_epochs):
        # Train epoch
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate_model(model, val_loader, criterion, device)
        val_losses.append(val_loss)

        # Evaluate on test set
        test_loss, mse, r2 = evaluate_model(model, test_loader, config.scaler_y, criterion, device)
        mse_scores.append(mse)
        r2_scores.append(r2)

        # Print progress
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch [{epoch + 1}/{config.num_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Test MSE: {mse:.4f}, Test R²: {r2:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            if save_model:
                save_model_checkpoint(
                    model, optimizer, epoch + 1, val_loss,
                    config.model_checkpoint_path
                )
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Training completed. Best validation loss: {best_val_loss:.4f}")
    return model, train_losses, val_losses, mse_scores, r2_scores


def save_model_checkpoint(
    model: NeuralNet,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str
):
    """
    Save model checkpoint including optimizer state.

    Args:
        model: Trained model
        optimizer: Optimizer
        epoch: Current epoch
        loss: Current loss
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Model checkpoint saved to {filepath}")


def load_model_checkpoint(
    filepath: str,
    model: NeuralNet,
    optimizer: Optional[optim.Optimizer] = None
) -> Tuple[NeuralNet, int, float]:
    """
    Load model checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)

    Returns:
        Tuple of (loaded_model, epoch, loss)
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Model checkpoint loaded from {filepath} (epoch {epoch}, loss {loss:.4f})")
    return model, epoch, loss


def load_model_for_inference(
    model_path: str,
    input_size: int,
    output_size: int,
    config,
    device: torch.device
) -> NeuralNet:
    """
    Load trained model for inference.

    Args:
        model_path: Path to saved model
        input_size: Number of input features
        output_size: Number of output targets
        config: Configuration object
        device: Device to load model on

    Returns:
        Loaded model in evaluation mode
    """
    model = NeuralNet(
        input_size=input_size,
        output_size=output_size,
        hidden_dim1=config.hidden_dim1,
        hidden_dim2=config.hidden_dim2,
        hidden_dim3=config.hidden_dim3,
        hidden_dim4=config.hidden_dim4,
        dropout_rate=config.dropout_rate
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    return model
