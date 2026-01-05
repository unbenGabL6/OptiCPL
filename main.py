"""
Main execution script for Stage 1 Model
Supports both training and inference modes
"""

import argparse
import os
import sys
import random
import numpy as np
import torch
import pandas as pd
import torch.nn as nn

from OptiCPL_refactored.config import Config
from OptiCPL_refactored.data_utils import (
    load_training_data, prepare_features, prepare_labels,
    create_dataloaders, load_scalers, save_scalers
)
from OptiCPL_refactored.model import create_model
from OptiCPL_refactored.trainer import train_model, evaluate_model, load_model_for_inference
from OptiCPL_refactored.inference import run_inference_from_file, save_predictions, batch_predict
from OptiCPL_refactored.visualization import (
    plot_training_curves, plot_prediction_comparison,
    plot_residuals, plot_error_distribution
)
from OptiCPL_refactored.train_logger import create_train_log_dir, save_training_info, find_log_dir_by_experiment
from OptiCPL_refactored.pipeline import run_end_to_end_pipeline
from OptiCPL_refactored.raw_data_processor import process_raw_data, save_intermediate_results, create_stage1_training_data, validate_raw_data
from OptiCPL_refactored.vae_model import load_vae_model
from OptiCPL_refactored.simclr_model import load_simclr_model


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_mode(config: Config):
    """
    Run model training pipeline.

    Args:
        config: Configuration object
    """
    print("=" * 60)
    print("Stage 1 Model - Training Mode")
    print("=" * 60)

    # Create training log directory
    print("\n[Step 0] Creating training log directory...")
    log_dir = create_train_log_dir(experiment_name=config.experiment_name)
    config.log_dir = log_dir
    config.setup_training_paths(log_dir)

    # Save training information
    print("\n[Step 0] Saving training configuration...")
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    save_training_info(log_dir, config_dict)

    # Randomly select seed from 29, 68, or 88
    FIXED_SEED = random.choice([29, 68, 88])
    set_random_seeds(FIXED_SEED)
    config.random_seed = FIXED_SEED

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load training data
    print("\n[Step 1] Loading training data...")
    df_train = load_training_data(config.training_data_path)
    print(f"   Loaded {len(df_train)} training samples")

    # Prepare features and labels
    print("\n[Step 2] Preparing features and labels...")
    X = prepare_features(df_train, config.img_column, config.glum_column, config.optical_column)
    y = prepare_labels(df_train, config.target_columns)
    print(f"   Feature shape: {X.shape}")
    print(f"   Label shape: {y.shape}")

    # Create dataloaders
    print("\n[Step 3] Creating dataloaders...")
    train_loader, val_loader, test_loader, scaler_x, scaler_y = create_dataloaders(
        X, y,
        test_size=config.test_size,
        val_size=config.val_size,
        batch_size=config.batch_size,
        random_seeds=(config.train_test_split_seed, config.val_split_seed),
        device=device
    )
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print(f"   Test batches: {len(test_loader)}")

    # Save scalers
    print("\n[Step 4] Saving scalers...")
    save_scalers(scaler_x, scaler_y, config.scaler_x_path, config.scaler_y_path)

    # Store scalers in config for later use
    config.scaler_x = scaler_x
    config.scaler_y = scaler_y

    # Create and initialize model
    print("\n[Step 5] Creating model...")
    input_size = X.shape[1]
    output_size = len(config.target_columns)
    model = create_model(input_size, output_size, config)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Input size: {input_size}, Output size: {output_size}")

    # Train model
    print("\n[Step 6] Training model...")
    model, train_losses, val_losses, mse_scores, r2_scores = train_model(
        model, train_loader, val_loader, test_loader, config, device
    )

    # Plot training curves
    print("\n[Step 7] Plotting training curves...")
    plot_training_curves(
        train_losses, val_losses, r2_scores,
        save_path=config.get_visualization_path("training_curves.png")
    )

    # Final evaluation
    print("\n[Step 8] Final evaluation...")
    test_loss, final_mse, final_r2 = evaluate_model(
        model, test_loader, scaler_y, nn.MSELoss(), device
    )
    print(f"   Final Test MSE: {final_mse:.4f}")
    print(f"   Final Test R²:  {final_r2:.4f}")

    # Get test predictions for visualization
    _, y_pred = batch_predict(model, test_loader, scaler_y, device)

    # Get actual values
    y_test_actual = []
    for _, y_batch in test_loader:
        y_test_actual.append(y_batch.cpu().numpy())
    y_test_actual = scaler_y.inverse_transform(np.concatenate(y_test_actual, axis=0))

    # Plot prediction comparison
    print("\n[Step 9] Plotting predictions vs actual...")
    plot_prediction_comparison(
        y_test_actual, y_pred, config.target_columns,
        save_path=config.get_visualization_path("prediction_comparison.png")
    )

    # Plot residuals
    print("\n[Step 10] Plotting residuals...")
    plot_residuals(
        y_test_actual, y_pred, config.target_columns,
        save_path=config.get_visualization_path("residuals.png")
    )

    # Plot error distribution
    print("\n[Step 11] Plotting error distribution...")
    plot_error_distribution(
        y_test_actual, y_pred, config.target_columns,
        save_path=config.get_visualization_path("error_distribution.png")
    )

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


def predict_mode(config: Config):
    """
    Run prediction/inference pipeline.

    Args:
        config: Configuration object
    """
    print("=" * 60)
    print("Stage 1 Model - Prediction Mode")
    print("=" * 60)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # If experiment name is provided and no custom model path is set,
    # automatically find the most recent training log with that experiment name
    if config.experiment_name and config.model_checkpoint_path == "/home/ylzhang/Nano_HF/SunHaifeng/Model/OptiCPL_refactored/stage1_best_model.pth":
        print(f"\nLooking for training log with experiment name: {config.experiment_name}")
        log_dir = find_log_dir_by_experiment(config.experiment_name)
        if log_dir:
            config.model_checkpoint_path = os.path.join(log_dir, "models", "stage1_best_model.pth")
            config.scaler_x_path = os.path.join(log_dir, "scalers", "scaler_x.pkl")
            config.scaler_y_path = os.path.join(log_dir, "scalers", "scaler_y.pkl")
            print(f"   Found log directory: {log_dir}")
            print(f"   Using model: {config.model_checkpoint_path}")
            print(f"   Using scalers from: {os.path.join(log_dir, 'scalers')}")
            # Also update predictions output path to be within the log directory
            config.predictions_output_path = os.path.join(log_dir, "results", "predictions_inference.csv")
        else:
            print(f"   WARNING: No training log found with experiment name '{config.experiment_name}'")

    # Infer input/output sizes if not provided
    print("\n1. Loading data...")
    df_train = load_training_data(config.training_data_path)
    X = prepare_features(df_train, config.img_column, config.glum_column, config.optical_column)
    y = prepare_labels(df_train, config.target_columns)
    input_size = X.shape[1]
    output_size = len(config.target_columns)
    print(f"   Inferred input size: {input_size}")
    print(f"   Inferred output size: {output_size}")

    # Run inference
    print("\n2. Running predictions...")
    predictions = run_inference_from_file(
        config.inference_data_path,
        config.model_checkpoint_path,
        input_size,
        output_size,
        config,
        device
    )
    print(f"   Generated predictions for {len(predictions)} samples")
    print(f"   Prediction shape: {predictions.shape}")

    # Save predictions
    print("\n3. Saving predictions...")
    save_predictions(
        predictions,
        config.predictions_output_path,
        config.target_columns
    )
    print(f"   Predictions saved to: {config.predictions_output_path}")

    # Print summary statistics
    print("\n4. Prediction Summary:")
    df_pred = pd.DataFrame(predictions, columns=config.target_columns)
    for col in config.target_columns:
        print(f"   {col:20s}: Mean={df_pred[col].mean():8.4f}, Std={df_pred[col].std():8.4f}")

    print("\n" + "=" * 60)
    print("Prediction completed successfully!")
    print("=" * 60)


def pipeline_mode(config: Config):
    """
    Run end-to-end pipeline: glum → features → fabrication parameters

    Args:
        config: Configuration object
    """
    # Randomly select seed from 29, 68, or 88
    # FIXED_SEED = random.choice([29, 68, 88])
    FIXED_SEED = 68
    set_random_seeds(FIXED_SEED)
    config.random_seed = FIXED_SEED

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Run pipeline
    run_end_to_end_pipeline(
        glum_data=config.glum_data_src_path,
        config=config,
        experiment_name=config.experiment_name,
        device=device
    )


def raw_pipeline_mode(config: Config):
    """
    Run raw data pipeline: raw data (with spectra) → features → fabrication parameters

    Args:
        config: Configuration object
    """
    print("=" * 80)
    print("OptiCPL Raw Data Pipeline")
    print("=" * 80)

    # Step 0: Create experiment log directory
    print("\n[Step 0] Creating experiment log directory...")
    from OptiCPL_refactored.train_logger import create_train_log_dir, save_training_info
    log_dir = create_train_log_dir(experiment_name=config.experiment_name)
    config.log_dir = log_dir
    config.setup_raw_pipeline_paths(log_dir)
    print(f"  Log directory: {log_dir}")

    # Save configuration
    config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('_')}
    save_training_info(log_dir, config_dict)
    print(f"  Configuration saved to training_info.txt")

    # Set random seed
    import random
    # FIXED_SEED = random.choice([29, 68, 88])
    FIXED_SEED = 4
    set_random_seeds(FIXED_SEED)
    config.random_seed = FIXED_SEED
    print(f"  Random seed: {FIXED_SEED}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Step 1: Process raw data
    print("\n[Step 1] Processing raw data...")
    output_dict = process_raw_data(
        file_path=config.raw_data_path,
        vae_model_path=config.raw_pipeline_vae_model_path,
        img_model_path=config.raw_pipeline_img_model_path,
        img_dir=config.raw_pipeline_img_dir,
        device=device,
        batch_size=config.batch_size
    )
    print("  Raw data processing completed")

    # Step 2: Save intermediate results
    print("\n[Step 2] Saving intermediate results...")
    save_intermediate_results(output_dict, log_dir)
    print("  Intermediate results saved")

    # Step 3: Prepare training data
    print("\n[Step 3] Preparing Stage1 training data...")
    df_train = create_stage1_training_data(output_dict)
    print(f"  Prepared {len(df_train)} training samples")

    # Step 4: Train Stage1 model
    print("\n[Step 4] Training Stage1 model...")
    from OptiCPL_refactored.data_utils import prepare_features, prepare_labels, create_dataloaders, save_scalers
    X = prepare_features(df_train, config.img_column, config.glum_column, config.optical_column)
    y = prepare_labels(df_train, config.target_columns)
    print(f"  Feature shape: {X.shape}")
    print(f"  Label shape: {y.shape}")

    # Create dataloaders
    train_loader, val_loader, test_loader, scaler_x, scaler_y = create_dataloaders(
        X, y,
        test_size=config.test_size,
        val_size=config.val_size,
        batch_size=config.batch_size,
        random_seeds=(config.train_test_split_seed, config.val_split_seed),
        device=device
    )
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Validation batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")

    # Save scalers
    save_scalers(scaler_x, scaler_y, config.scaler_x_path, config.scaler_y_path)
    config.scaler_x = scaler_x
    config.scaler_y = scaler_y
    print("  Scalers saved")

    # Create model
    input_size = X.shape[1]
    output_size = len(config.target_columns)
    model = create_model(input_size, output_size, config)
    print(f"  Model created: {input_size} -> {output_size}")

    # Train model
    model, train_losses, val_losses, mse_scores, r2_scores = train_model(
        model, train_loader, val_loader, test_loader, config, device
    )

    # Step 5: Evaluate model
    print("\n[Step 5] Evaluating model...")
    from OptiCPL_refactored.trainer import evaluate_model
    from OptiCPL_refactored.inference import batch_predict
    test_loss, final_mse, final_r2 = evaluate_model(
        model, test_loader, scaler_y, nn.MSELoss(), device
    )
    print(f"  Test MSE: {final_mse:.4f}")
    print(f"  Test R²:  {final_r2:.4f}")

    # Get predictions for visualization
    _, y_pred = batch_predict(model, test_loader, scaler_y, device)
    y_test_actual = []
    for _, y_batch in test_loader:
        y_test_actual.append(y_batch.cpu().numpy())
    y_test_actual = scaler_y.inverse_transform(np.concatenate(y_test_actual, axis=0))

    # Step 6: Plot results
    print("\n[Step 6] Generating visualizations...")
    from OptiCPL_refactored.visualization import plot_training_curves, plot_prediction_comparison, plot_residuals, plot_error_distribution
    plot_training_curves(
        train_losses, val_losses, r2_scores,
        save_path=config.get_visualization_path("training_curves.png")
    )
    plot_prediction_comparison(
        y_test_actual, y_pred, config.target_columns,
        save_path=config.get_visualization_path("prediction_comparison.png")
    )
    plot_residuals(
        y_test_actual, y_pred, config.target_columns,
        save_path=config.get_visualization_path("residuals.png")
    )
    plot_error_distribution(
        y_test_actual, y_pred, config.target_columns,
        save_path=config.get_visualization_path("error_distribution.png")
    )
    print("  Visualizations saved")

    print("\n" + "=" * 80)
    print("Raw pipeline completed successfully!")
    print(f"Results saved to: {log_dir}")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Stage 1 Model: Predicts fabrication parameters from material features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python main.py train --epochs 2000 --lr 0.001

  # Train with experiment name (creates timestamped log directory)
  python main.py train --experiment-name "hyperparam_search_1"

  # Make predictions with existing model
  python main.py predict --input data.csv --output predictions.csv

  # Train with custom config
  python main.py train --config my_config.py
        """
    )

    parser.add_argument(
        "mode",
        choices=["train", "predict", "pipeline", "raw-pipeline"],
        help="Mode: train a new model, predict from CSV, end-to-end pipeline from glum, or raw data pipeline"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config.py",
        help="Path to configuration file (default: config.py)"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Path to model checkpoint (required for predict mode)"
    )

    parser.add_argument(
        "--input",
        type=str,
        help="Path to input data for predictions"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Path to save predictions"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of training epochs"
    )

    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to use (auto/cpu/cuda)"
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        help="Experiment name for organizing training runs"
    )


    parser.add_argument(
        "--glum-data",
        type=str,
        help="Path to glum data for pipeline mode"
    )

    parser.add_argument(
        "--raw-data",
        type=str,
        help="Path to raw data for raw-pipeline mode (Data_complete_corrected_v7.xlsx)"
    )

    parser.add_argument(
        "--img-dir",
        type=str,
        help="Directory containing SEM images for raw-pipeline mode"
    )

    parser.add_argument(
        "--vae-model",
        type=str,
        help="Path to VAE model for Optical feature generation in raw-pipeline mode"
    )

    parser.add_argument(
        "--img-model",
        type=str,
        help="Path to SimCLR model for IMG feature generation in raw-pipeline mode"
    )

    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Override config from command line arguments
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.input:
        config.inference_data_path = args.input
    if args.output:
        config.predictions_output_path = args.output
    if args.model:
        config.model_checkpoint_path = args.model
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.raw_data:
        config.raw_data_path = args.raw_data
    if args.img_dir:
        config.raw_pipeline_img_dir = args.img_dir
    if args.vae_model:
        config.raw_pipeline_vae_model_path = args.vae_model
    if args.img_model:
        config.raw_pipeline_img_model_path = args.img_model

    # Set device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    config.device = device  # Store device in config

    # Print configuration
    print("Configuration:")
    print(f"  Random seed: {config.random_seed}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Model save dir: {config.model_save_dir}")

    # Run selected mode
    if args.mode == "train":
        train_mode(config)
    elif args.mode == "predict":
        predict_mode(config)
    elif args.mode == "pipeline":
        pipeline_mode(config)
    elif args.mode == "raw-pipeline":
        raw_pipeline_mode(config)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
