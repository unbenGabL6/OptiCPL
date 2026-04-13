"""
Stage 2 Inference Module for OptiCPL
Handles generating features (IMG and Optical) from glum data
"""

import re
import numpy as np
import pandas as pd
import torch
from typing import Union, List, Tuple
import warnings

from .stage2_models import Stage2ModelManager


def parse_glum_data(glum_data: Union[pd.DataFrame, List, np.ndarray]) -> np.ndarray:
    """
    Parse glum data from various formats into numpy array

    Args:
        glum_data: Input glum data in various formats:
                   - pandas DataFrame with 'glum' column
                   - List of strings: ["0.1 -0.2 3.4 ...", ...]
                   - List of lists: [[0.1, -0.2, 3.4, ...], ...]
                   - numpy array

    Returns:
        numpy array of shape (n_samples, 40)
    """
    # Handle DataFrame
    if isinstance(glum_data, pd.DataFrame):
        if 'glum' in glum_data.columns:
            glum_data = glum_data['glum'].tolist()
        else:
            raise ValueError("DataFrame must contain 'glum' column")

    parsed_data = []

    for item in glum_data:
        # Handle string (space-separated values)
        if isinstance(item, str):
            values = re.findall(r'-?\d+\.?\d*', item)
            parsed = [float(v) for v in values]
        # Handle list/tuple/ndarray
        elif isinstance(item, (list, tuple, np.ndarray)):
            parsed = [float(v) for v in item]
        else:
            raise ValueError(f"Unsupported data type: {type(item)}")

        # Validate dimension
        if len(parsed) != 40:
            raise ValueError(f"Expected 40 features, got {len(parsed)}")

        parsed_data.append(parsed)

    return np.array(parsed_data)


def predict_optical(glum_data: Union[pd.DataFrame, List, np.ndarray],
                    model_manager: Stage2ModelManager,
                    device: str = 'cpu',
                    batch_size: int = 256) -> np.ndarray:
    """

    Args:
        glum_data: Input glum data (various formats)
        model_manager: Stage2ModelManager instance
        device: Device for computation ('cpu' or 'cuda')
        batch_size: Batch size for prediction

    Returns:
        Predicted Optical features (n_samples, 32)
    """
    model, scaler_X, scaler_Y = model_manager.load_optical_model(device)

    # Parse input data
    glum_matrix = parse_glum_data(glum_data)

    # Apply scaling
    glum_scaled = scaler_X.transform(glum_matrix)

    # Convert to tensor
    inputs = torch.tensor(glum_scaled, dtype=torch.float32, device=device)

    # Predict in batches
    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            batch_pred = model(batch)
            predictions.append(batch_pred.cpu().numpy())

    # Concatenate predictions
    pred_scaled = np.vstack(predictions)

    # Inverse transform to original scale
    predictions_actual = scaler_Y.inverse_transform(pred_scaled)

    return predictions_actual


def predict_img(glum_data: Union[pd.DataFrame, List, np.ndarray],
                model_manager: Stage2ModelManager,
                device: str = 'cpu',
                batch_size: int = 256) -> np.ndarray:
    """
    Args:
        glum_data: Input glum data (various formats)
        model_manager: Stage2ModelManager instance
        device: Device for computation ('cpu' or 'cuda')
        batch_size: Batch size for prediction

    Returns:
        Predicted IMG features (n_samples, 128)
    """
    model, scaler_X, scaler_Y = model_manager.load_img_model(device)

    # Parse input data
    glum_matrix = parse_glum_data(glum_data)

    # Apply scaling
    glum_scaled = scaler_X.transform(glum_matrix)

    # Convert to tensor
    inputs = torch.tensor(glum_scaled, dtype=torch.float32, device=device)

    # Predict in batches
    predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i+batch_size]
            batch_pred = model(batch)
            predictions.append(batch_pred.cpu().numpy())

    # Concatenate predictions
    pred_scaled = np.vstack(predictions)

    # Inverse transform to original scale
    predictions_actual = scaler_Y.inverse_transform(pred_scaled)

    return predictions_actual


def generate_stage2_features(glum_data: Union[pd.DataFrame, List, np.ndarray],
                             model_manager: Stage2ModelManager,
                             device: str = 'cpu',
                             batch_size: int = 256) -> pd.DataFrame:
    """
    Generate complete stage2 features from glum data

    Args:
        glum_data: Input glum data (various formats)
        model_manager: Stage2ModelManager instance
        device: Device for computation
        batch_size: Batch size for prediction

    Returns:
        DataFrame with glum, IMG, and Optical features
    """
    # Parse glum data
    glum_matrix = parse_glum_data(glum_data)

    print("Predicting Optical features...")
    optical_features = predict_optical(glum_data, model_manager, device, batch_size)

    print("Predicting IMG features...")
    img_features = predict_img(glum_data, model_manager, device, batch_size)

    # Create DataFrame
    # If input was DataFrame, preserve index and other columns
    if isinstance(glum_data, pd.DataFrame):
        df = glum_data.copy()
    else:
        df = pd.DataFrame({'glum': [str(x) for x in glum_matrix.tolist()]})

    # Add parsed glum data
    df['glum_parsed'] = glum_matrix.tolist()

    # Add Optical features
    df['Optical'] = optical_features.tolist()

   
    df['IMG_seed28'] = img_features.tolist()

    return df


def save_stage2_features(features_df: pd.DataFrame, output_path: str):
    """
    Save stage2 features to CSV file

    Args:
        features_df: DataFrame with generated features
        output_path: Path to save CSV file
    """
    # Ensure output directory exists
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    features_df.to_csv(output_path, index=False)
    print(f"Stage2 features saved to: {output_path}")


# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
