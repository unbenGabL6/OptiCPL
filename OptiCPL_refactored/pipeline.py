"""
Pipeline Module for OptiCPL
Handles end-to-end pipeline: glum → features → fabrication parameters
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from typing import Union, List, Optional

from .config import Config
from .stage2_inference import generate_stage2_features, parse_glum_data
from .stage2_models import Stage2ModelManager
from .inference import run_inference_from_dataframe, save_predictions
from .trainer import load_model_for_inference, train_model
from .train_logger import create_train_log_dir, save_training_info
import warnings

# Suppress sklearn version warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def save_stage2_artifacts(log_dir: str,
                          model_manager: Stage2ModelManager,
                          device: str = 'cpu'):
    """
    Save stage2 models and scalers to log directory

    Args:
        log_dir: Log directory path
        model_manager: Stage2ModelManager instance
        device: Device to use
    """
    # Create directories
    models_dir = Path(log_dir) / "models" / "stage2"
    scalers_dir = Path(log_dir) / "scalers" / "stage2"
    models_dir.mkdir(parents=True, exist_ok=True)
    scalers_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[Stage2] Saving artifacts to {log_dir}")

    # Save Optical model and scalers (seed 13)
    print("  Saving Optical model (seed 13)...")
    optical_model, optical_scaler_X, optical_scaler_Y = model_manager.load_optical_model(device)

    # Save model state dict
    torch.save(optical_model.state_dict(), models_dir / "optical_model.pth")

    # Save scalers
    with open(scalers_dir / "optical_scaler_X.pkl", "wb") as f:
        pickle.dump(optical_scaler_X, f)
    with open(scalers_dir / "optical_scaler_Y.pkl", "wb") as f:
        pickle.dump(optical_scaler_Y, f)

    # Save IMG model and scalers (seed 28)
    print("  Saving IMG model (seed 28)...")
    img_model, img_scaler_X, img_scaler_Y = model_manager.load_img_model(device)

    # Save model state dict
    torch.save(img_model.state_dict(), models_dir / "img_model.pth")

    # Save scalers
    with open(scalers_dir / "img_scaler_X.pkl", "wb") as f:
        pickle.dump(img_scaler_X, f)
    with open(scalers_dir / "img_scaler_Y.pkl", "wb") as f:
        pickle.dump(img_scaler_Y, f)

    print("  ✓ Stage2 artifacts saved successfully")


def run_end_to_end_pipeline(glum_data: Union[str, pd.DataFrame, List, np.ndarray],
                           config: Config,
                           experiment_name: Optional[str] = None,
                           device: str = 'cpu',
                           batch_size: int = 256) -> np.ndarray:
    """
    Run complete pipeline: glum → features → fabrication parameters

    Args:
        glum_data: Glum input data (file path, DataFrame, list, or array)
        config: Configuration object
        experiment_name: Optional experiment name for log directory
        device: Device for computation
        batch_size: Batch size for predictions

    Returns:
        Predicted fabrication parameters
    """
    print("=" * 70)
    print("OptiCPL End-to-End Pipeline")
    print("=" * 70)

    # Create log directory
    print("\n[Step 1] Creating pipeline log directory...")
    log_dir = create_train_log_dir(
        base_dir=None,
        experiment_name=experiment_name if experiment_name else "pipeline"
    )
    config.log_dir = log_dir
    config.setup_pipeline_paths(log_dir)
    print(f"  Log directory: {log_dir}")

    # Save configuration
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    save_training_info(log_dir, config_dict)
    print("  Configuration saved to training_info.txt")

    # Load glum data
    print("\n[Step 2] Loading glum data...")
    if isinstance(glum_data, str):
        # Load from file path
        if glum_data.endswith('.xlsx'):
            glum_df = pd.read_excel(glum_data)
        elif glum_data.endswith('.csv'):
            glum_df = pd.read_csv(glum_data)
        else:
            raise ValueError(f"Unsupported file format: {glum_data}")
    elif isinstance(glum_data, pd.DataFrame):
        glum_df = glum_data
    else:
        # Convert list/array to DataFrame
        glum_matrix = parse_glum_data(glum_data)
        glum_df = pd.DataFrame({'glum': [str(x) for x in glum_matrix.tolist()]})

    print(f"  Loaded {len(glum_df)} samples")

    # Initialize stage2 model manager
    print("\n[Step 3] Initializing Stage2 models...")
    model_manager = Stage2ModelManager(
        img_model_src_dir=config.stage2_img_model_src_dir,
        img_scaler_src_dir=config.stage2_img_scaler_src_dir,
        optical_model_src_dir=config.stage2_optical_model_src_dir,
        optical_scaler_src_dir=config.stage2_optical_scaler_src_dir
    )
    print("  Stage2ModelManager initialized")

    # Save stage2 artifacts to log directory
    save_stage2_artifacts(log_dir, model_manager, device)

    # Generate stage2 features
    print("\n[Step 4] Generating Stage2 features...")
    features_df = generate_stage2_features(
        glum_df,
        model_manager,
        device=device,
        batch_size=batch_size
    )
    print(f"  Generated features for {len(features_df)} samples")
    print(f"  Feature dimensions: IMG=128, glum=40, Optical=32")

    # Save features to CSV
    features_output_path = config.stage2_features_output_path
    features_df.to_csv(features_output_path, index=False)
    print(f"  ✓ Features saved to: {features_output_path}")

    # Load training data to infer input/output sizes for stage1
    print("\n[Step 5] Preparing Stage1 model...")
    from .data_utils import load_training_data, prepare_features, prepare_labels, create_dataloaders, save_scalers
    from .model import NeuralNet, create_model

    df_train = load_training_data(config.training_data_path)
    X = prepare_features(df_train, config.img_column, config.glum_column, config.optical_column)
    y = prepare_labels(df_train, config.target_columns)

    input_size = X.shape[1]
    output_size = len(config.target_columns)
    print(f"  Stage1 input size: {input_size}")
    print(f"  Stage1 output size: {output_size}")

    # Create dataloaders for training
    print("\n[Step 6] Creating Stage1 training dataloaders...")
    train_loader, val_loader, test_loader, scaler_x, scaler_y = create_dataloaders(
        X, y,
        test_size=config.test_size,
        val_size=config.val_size,
        batch_size=config.batch_size,
        random_seeds=(config.train_test_split_seed, config.val_split_seed),
        device=device
    )
    print(f"    Training batches: {len(train_loader)}")
    print(f"    Validation batches: {len(val_loader)}")
    print(f"    Test batches: {len(test_loader)}")

    # Save scalers
    print("\n[Step 7] Saving Stage1 scalers...")
    save_scalers(scaler_x, scaler_y, config.scaler_x_path, config.scaler_y_path)

    # Store scalers in config for training
    config.scaler_x = scaler_x
    config.scaler_y = scaler_y

    # Create model
    print("\n[Step 8] Creating Stage1 model...")
    model = create_model(input_size, output_size, config)
    print(f"    Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train model
    print("\n[Step 9] Training Stage1 model...")
    model, train_losses, val_losses, _, val_r2_scores = train_model(
        model, train_loader, val_loader, test_loader, config, device, save_model=True
    )
    print(f"    Training completed with best validation R²: {max(val_r2_scores):.4f}")

    # Run inference using newly trained model
    print("\n[Step 10] Running Stage1 inference with newly trained model...")
    from .data_utils import prepare_inference_features

    # Use the newly trained model directly (not from checkpoint)
    model.eval()
    with torch.no_grad():
        # Prepare input features - extract and combine only numeric feature columns
        # Note: IMG features are in 'IMG_seed28' column from Stage2
        features_tensor = prepare_inference_features(
            features_df, scaler_x,
            img_column=config.img_column,
            img_seed_column='IMG_seed28',  # Stage2 generates IMG features in this column
            glum_column='glum_parsed',     # Use parsed glum data
            optical_column=config.optical_column
        ).to(device)

        # Run prediction
        predictions_scaled = model(features_tensor).cpu().numpy()
        predictions = scaler_y.inverse_transform(predictions_scaled)

    print(f"  Generated {len(predictions)} predictions")

    # Save predictions
    predictions_output_path = config.pipeline_predictions_output_path
    save_predictions(predictions, predictions_output_path, config.target_columns)
    print(f"  ✓ Predictions saved to: {predictions_output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Pipeline completed successfully!")
    print("=" * 70)
    print(f"Results location: {log_dir}")
    print(f"  - Features: {features_output_path}")
    print(f"  - Predictions: {predictions_output_path}")
    print(f"  - Training info: {log_dir}/training_info.txt")
    print("=" * 70)

    return predictions
