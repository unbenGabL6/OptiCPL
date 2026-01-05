"""
Raw Data Processor for OptiCPL Pipeline
Handles processing of raw data (Data_complete_corrected_v7.xlsx) to generate features
"""

import pandas as pd
import numpy as np
import os
import ast
from typing import Tuple, Dict, Optional
import torch

from .vae_model import load_vae_model, extract_optical_features, min_max_scale
from .simclr_model import load_simclr_model, extract_img_features_for_numbers


def load_raw_data(file_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, list]:
    """
    Load raw data from Excel file

    Args:
        file_path: Path to Data_complete_corrected_v7.xlsx

    Returns:
        Tuple of (df, glum_data, numbers, valid_indices)
    """
    # Read Excel file
    df = pd.read_excel(file_path)

    # Extract number column
    numbers = df['number'].tolist()

    # Extract glum data
    def safe_eval(val):
        try:
            if isinstance(val, str):
                return ast.literal_eval(val)
            return val
        except (ValueError, SyntaxError):
            return None

    def adjust_glum_length(glum_list):
        """Adjust glum list to 40 dimensions"""
        if glum_list is None:
            return None

        length = len(glum_list)

        if length == 40:
            return glum_list
        elif length == 41:
            # Remove last element
            return glum_list[:-1]
        elif length == 61:
            # Remove every 3rd element and last element
            return [glum_list[i] for i in range(len(glum_list) - 1) if i % 3 != 2]
        elif length == 81:
            # Keep every 2nd element (even indices) and remove last
            return [glum_list[i] for i in range(0, len(glum_list) - 1, 2)]
        else:
            # For other lengths, try to interpolate or use first 40
            print(f"Warning: Unexpected glum length {length}, using first 40 elements")
            return glum_list[:40]

    # Parse and adjust glum data
    glum_parsed = df['glum'].apply(safe_eval).dropna()
    glum_adjusted = glum_parsed.apply(adjust_glum_length).dropna()

    valid_indices = glum_adjusted.index.tolist()

    # Filter DataFrame to only include valid rows
    df_valid = df.loc[valid_indices].reset_index(drop=True)
    numbers_valid = [numbers[i] for i in valid_indices]
    glum_data = np.vstack(glum_adjusted.values)

    # Verify all glum data is 40-dimensional
    assert glum_data.shape[1] == 40, f"Expected glum dimension 40, got {glum_data.shape[1]}"

    return df_valid, glum_data, numbers_valid, valid_indices


def parse_spectral_columns(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Parse spectral data columns from DataFrame

    Args:
        df: DataFrame containing spectral columns

    Returns:
        Dictionary mapping column names to numpy arrays
    """
    def safe_eval(val):
        try:
            if isinstance(val, str):
                return ast.literal_eval(val)
            return val
        except (ValueError, SyntaxError):
            return None

    spectral_data = {}
    spectral_columns = ['PL', 'EX_sample', 'CD_sample', 'CD_template', 'EX_template']

    for col in spectral_columns:
        if col in df.columns:
            parsed_list = df[col].apply(safe_eval).dropna().tolist()
            spectral_data[col] = np.array(parsed_list, dtype=np.float32)
        else:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    return spectral_data


def validate_raw_data(df: pd.DataFrame, img_dir: str, numbers: list) -> Dict[str, any]:
    """
    Validate raw data and check for missing images

    Args:
        df: DataFrame containing raw data
        img_dir: Directory containing SEM images
        numbers: List of sample numbers

    Returns:
        Validation report
    """
    report = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    # Check required columns
    required_cols = ['number', 'glum', 'PL', 'EX_sample', 'CD_sample', 'CD_template', 'EX_template']
    for col in required_cols:
        if col not in df.columns:
            report['valid'] = False
            report['errors'].append(f"Missing required column: {col}")

    # Check for missing images
    missing_images = []
    for number in numbers:
        for suffix in ['1', '3', '5']:
            img_path = os.path.join(img_dir, f"{number}_{suffix}.tif")
            if not os.path.exists(img_path):
                missing_images.append(img_path)

    if missing_images:
        report['warnings'].append(f"Missing {len(missing_images)} images")
        # Show first 5 missing images
        for img in missing_images[:5]:
            report['warnings'].append(f"  - {img}")

    # Data stats
    report['stats'] = {
        'num_samples': len(df),
        'missing_images_count': len(missing_images),
        'glum_dim': len(df['glum'].iloc[0]) if len(df) > 0 else 0
    }

    return report


def process_raw_data(file_path: str, vae_model_path: str, img_model_path: str,
                     img_dir: str, device: str = 'cpu', batch_size: int = 32) -> Dict[str, any]:
    """
    Main function to process raw data and generate all features

    Args:
        file_path: Path to raw data Excel file
        vae_model_path: Path to VAE model
        img_model_path: Path to SimCLR model
        img_dir: Directory containing SEM images
        device: Device to run inference on
        batch_size: Batch size for inference

    Returns:
        Dictionary containing all generated features and metadata
    """
    print("=" * 80)
    print("Raw Data Processing Pipeline")
    print("=" * 80)

    # Step 1: Load raw data
    print("\n[Step 1] Loading raw data...")
    df, glum_data, numbers, valid_indices = load_raw_data(file_path)
    print(f"  Loaded {len(df)} valid samples")

    # Step 2: Validate data
    print("\n[Step 2] Validating data...")
    validation_report = validate_raw_data(df, img_dir, numbers)
    print(f"  Samples: {validation_report['stats']['num_samples']}")
    print(f"  Missing images: {validation_report['stats']['missing_images_count']}")

    if not validation_report['valid']:
        raise ValueError(f"Data validation failed: {validation_report['errors']}")

    # Step 3: Generate Optical features using VAE
    print("\n[Step 3] Generating Optical features...")
    vae_model = load_vae_model(vae_model_path, device)
    optical_features = extract_optical_features(vae_model, df, device, batch_size)
    print(f"  Generated Optical features: {optical_features.shape}")

    # Step 4: Generate IMG features using SimCLR
    print("\n[Step 4] Generating IMG features...")
    img_model = load_simclr_model(img_model_path, device)
    img_features = extract_img_features_for_numbers(img_model, numbers, img_dir, device)
    print(f"  Generated IMG features: {img_features.shape}")

    # Step 5: Prepare combined features
    print("\n[Step 5] Preparing combined features...")
    combined_features = np.hstack([img_features, glum_data, optical_features])
    print(f"  Combined features shape: {combined_features.shape}")
    print(f"    - IMG: {img_features.shape[1]} dim")
    print(f"    - glum: {glum_data.shape[1]} dim")
    print(f"    - Optical: {optical_features.shape[1]} dim")

    # Step 6: Create output dictionary
    print("\n[Step 6] Creating output dictionary...")
    output = {
        'df': df,
        'numbers': numbers,
        'glum_data': glum_data,
        'optical_features': optical_features,
        'img_features': img_features,
        'combined_features': combined_features,
        'validation_report': validation_report,
        'metadata': {
            'num_samples': len(df),
            'glum_dim': glum_data.shape[1],
            'optical_dim': optical_features.shape[1],
            'img_dim': img_features.shape[1],
            'combined_dim': combined_features.shape[1],
            'device': device
        }
    }

    print("\n" + "=" * 80)
    print("Raw data processing completed successfully!")
    print("=" * 80)

    return output


def save_intermediate_results(output_dict: Dict[str, any], log_dir: str):
    """
    Save all intermediate results to files

    Args:
        output_dict: Dictionary containing all features and data
        log_dir: Directory to save results
    """
    print("\n[Step 7] Saving intermediate results...")

    # Create directories
    features_dir = os.path.join(log_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)

    # Save glum data
    glum_path = os.path.join(features_dir, 'glum_parsed.csv')
    glum_df = pd.DataFrame(
        output_dict['glum_data'],
        columns=[f'glum_{i}' for i in range(output_dict['glum_data'].shape[1])]
    )
    glum_df['number'] = output_dict['numbers']
    glum_df.to_csv(glum_path, index=False)
    print(f"  Saved glum data: {glum_path}")

    # Save Optical features
    optical_path = os.path.join(features_dir, 'optical_features.csv')
    optical_df = pd.DataFrame(
        output_dict['optical_features'],
        columns=[f'optical_{i}' for i in range(output_dict['optical_features'].shape[1])]
    )
    optical_df['number'] = output_dict['numbers']
    optical_df.to_csv(optical_path, index=False)
    print(f"  Saved Optical features: {optical_path}")

    # Save IMG features
    img_path = os.path.join(features_dir, 'img_features.csv')
    img_df = pd.DataFrame(
        output_dict['img_features'],
        columns=[f'img_{i}' for i in range(output_dict['img_features'].shape[1])]
    )
    img_df['number'] = output_dict['numbers']
    img_df.to_csv(img_path, index=False)
    print(f"  Saved IMG features: {img_path}")

    # Save combined features (for Stage1 training)
    combined_path = os.path.join(features_dir, 'combined_features.csv')
    combined_df = pd.DataFrame(
        output_dict['combined_features'],
        columns=[f'feature_{i}' for i in range(output_dict['combined_features'].shape[1])]
    )
    combined_df['number'] = output_dict['numbers']
    combined_df.to_csv(combined_path, index=False)
    print(f"  Saved combined features: {combined_path}")

    # Save metadata
    metadata_path = os.path.join(log_dir, 'metadata.json')
    import json
    metadata = output_dict['metadata'].copy()
    # Convert device to string for JSON serialization
    if 'device' in metadata:
        metadata['device'] = str(metadata['device'])
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")

    print(f"\n  All results saved to: {log_dir}")


def create_stage1_training_data(output_dict: Dict[str, any]) -> pd.DataFrame:
    """
    Create DataFrame for Stage1 training

    Args:
        output_dict: Dictionary from process_raw_data

    Returns:
        DataFrame with columns: features, targets, number
    """
    df = output_dict['df'].copy()

    # Add features as string columns (for compatibility with existing data_utils)
    # glum (40-dim)
    glum_data = output_dict['glum_data']
    df['glum'] = [str(x.tolist()) for x in glum_data]

    # Optical (32-dim)
    optical_features = output_dict['optical_features']
    df['Optical'] = [str(x.tolist()) for x in optical_features]

    # IMG (128-dim)
    img_features = output_dict['img_features']
    df['IMG'] = [str(x.tolist()) for x in img_features]

    return df
