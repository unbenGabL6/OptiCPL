"""
Data utilities module for Stage 1 Model
Handles data loading, preprocessing, and DataLoader creation
"""

import ast
import pickle
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def parse_array_string(s: str) -> np.ndarray:
    """
    Parse string representation of array into NumPy array.

    Handles strings like '[ 2.2849341e-01 -2.0850410e+00 ...]' or
    '[-0.0539296, -0.0290061, ...]' and converts them to NumPy arrays.

    Args:
        s: String representation of an array

    Returns:
        NumPy array parsed from the string
    """
    if pd.isna(s):
        return np.array([])

    s = str(s).strip()
    if s.startswith('[') and s.endswith(']'):
        s = s[1:-1]

    # Handle comma-separated values by replacing commas with spaces
    s = s.replace(',', ' ')

    arr = np.fromstring(s, sep=' ')
    return arr


def safe_eval(val) -> Optional:
    """
    Safely evaluate string values using ast.literal_eval.

    Args:
        val: Value to evaluate (string or other type)

    Returns:
        Evaluated value if successful, None if parsing fails
    """
    try:
        if isinstance(val, str):
            return ast.literal_eval(val)
        return val
    except (ValueError, SyntaxError):
        return None


def load_training_data(file_path: str) -> pd.DataFrame:
    """
    Load training data from Excel file.

    Args:
        file_path: Path to Excel file

    Returns:
        Loaded DataFrame
    """
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        print(f"Error loading training data: {e}")
        raise


def load_inference_data(file_path: str) -> pd.DataFrame:
    """
    Load inference data from CSV file.

    Args:
        file_path: Path to CSV file

    Returns:
        Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error loading inference data: {e}")
        raise


def prepare_features(
    df: pd.DataFrame,
    img_column: str = "IMG",
    glum_column: str = "glum",
    optical_column: str = "Optical"
) -> np.ndarray:
    """
    Extract and combine feature matrices from DataFrame.

    Args:
        df: DataFrame containing feature columns
        img_column: Name of image feature column
        glum_column: Name of glum feature column
        optical_column: Name of optical feature column

    Returns:
        Combined feature matrix
    """
    # Parse array strings
    df[img_column] = df[img_column].apply(parse_array_string)
    df[glum_column] = df[glum_column].apply(safe_eval)
    df[optical_column] = df[optical_column].apply(safe_eval)

    # Stack matrices
    img_matrix = np.vstack(df[img_column].values)
    glum_matrix = np.vstack(df[glum_column].values)
    optical_matrix = np.vstack(df[optical_column].values)

    # Combine features horizontally
    X = np.hstack([img_matrix, glum_matrix, optical_matrix])

    return X


def prepare_labels(df: pd.DataFrame, target_columns: List[str]) -> np.ndarray:
    """
    Extract label matrix from DataFrame.

    Args:
        df: DataFrame containing target columns
        target_columns: List of target column names

    Returns:
        Label matrix
    """
    return df[target_columns].values


def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.25,
    batch_size: int = 32,
    random_seeds: Tuple[int, int] = (42, 42),
    device: torch.device = torch.device("cpu")
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler]:
    """
    Create train, validation, and test DataLoaders.

    Args:
        X: Feature matrix
        y: Label matrix
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation
        batch_size: Batch size for DataLoaders
        random_seeds: Tuple of (train_test_seed, val_split_seed)
        device: Device to move tensors to

    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler_x, scaler_y)
    """
    # Initialize scalers
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    # Fit and transform data
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    # Split data
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_scaled, y_scaled,
        test_size=test_size,
        random_state=random_seeds[0]
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size,
        random_state=random_seeds[1]
    )

    # Convert to tensors and move to device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler_x, scaler_y


def load_scalers(scaler_x_path: str, scaler_y_path: str) -> Tuple[StandardScaler, StandardScaler]:
    """
    Load saved StandardScalers from pickle files.

    Args:
        scaler_x_path: Path to scaler_x pickle file
        scaler_y_path: Path to scaler_y pickle file

    Returns:
        Tuple of (scaler_x, scaler_y)
    """
    try:
        with open(scaler_x_path, "rb") as f:
            scaler_x = pickle.load(f)

        with open(scaler_y_path, "rb") as f:
            scaler_y = pickle.load(f)

        return scaler_x, scaler_y
    except Exception as e:
        print(f"Error loading scalers: {e}")
        raise


def save_scalers(scaler_x: StandardScaler, scaler_y: StandardScaler,
                 scaler_x_path: str, scaler_y_path: str):
    """
    Save StandardScalers to pickle files.

    Args:
        scaler_x: Feature scaler
        scaler_y: Label scaler
        scaler_x_path: Path to save scaler_x
        scaler_y_path: Path to save scaler_y
    """
    try:
        with open(scaler_x_path, "wb") as f:
            pickle.dump(scaler_x, f)

        with open(scaler_y_path, "wb") as f:
            pickle.dump(scaler_y, f)

        print(f"Scalers saved to {scaler_x_path} and {scaler_y_path}")
    except Exception as e:
        print(f"Error saving scalers: {e}")
        raise


def prepare_inference_features(
    df: pd.DataFrame,
    scaler_x: StandardScaler,
    img_column: str = "IMG",
    img_seed_column: str = "IMG_seed28",
    glum_column: str = "glum",
    optical_column: str = "Optical"
) -> torch.Tensor:
    """
    Prepare features for inference/prediction.

    Args:
        df: DataFrame containing inference data
        scaler_x: Fitted StandardScaler for features
        img_column: Name of image feature column
        img_seed_column: Name of image feature column for inference (may differ)
        glum_column: Name of glum feature column
        optical_column: Name of optical feature column

    Returns:
        Scaled feature tensor ready for inference
    """
    # Use img_seed_column if it exists, otherwise use img_column
    actual_img_column = img_seed_column if img_seed_column in df.columns else img_column

    # Parse array strings
    df[actual_img_column] = df[actual_img_column].apply(safe_eval)
    df[glum_column] = df[glum_column].apply(safe_eval)
    df[optical_column] = df[optical_column].apply(safe_eval)

    # Stack matrices
    img_matrix = np.vstack(df[actual_img_column].values)
    glum_matrix = np.vstack(df[glum_column].values)
    optical_matrix = np.vstack(df[optical_column].values)

    # Combine features
    X = np.hstack([img_matrix, glum_matrix, optical_matrix])

    # Scale features
    X_scaled = scaler_x.transform(X)

    # Convert to tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    return X_tensor
