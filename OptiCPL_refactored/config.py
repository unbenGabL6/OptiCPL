"""
Configuration module for Stage 1 Model
Centralized configuration management for all hyperparameters and file paths
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Config:
    """Configuration class for Stage 1 Model training and inference"""

    # Random seeds for reproducibility
    random_seed: int = 400
    train_test_split_seed: int = 42
    val_split_seed: int = 42

    # File paths
    training_data_path: str = "/home/ylzhang/Nano_HF/OptiCPL/Data/DF_IMG.xlsx"
    inference_data_path: str = "/home/ylzhang/Nano_HF/OptiCPL/Results/predicted_features.csv"

    # Log directory (if None, will be generated dynamically during training)
    log_dir: Optional[str] = None

    # Experiment name for organizing training runs
    experiment_name: Optional[str] = None

    # Model save paths (relative to log_dir if provided, otherwise use absolute paths below)
    model_save_dir: str = "/home/ylzhang/Nano_HF/SunHaifeng/Model/OptiCPL_refactored"
    model_checkpoint_path: str = "/home/ylzhang/Nano_HF/SunHaifeng/Model/OptiCPL_refactored/stage1_best_model.pth"

    # Scaler save paths
    scaler_x_path: str = "/home/ylzhang/Nano_HF/SunHaifeng/Model/Scaler/scaler_x.pkl"
    scaler_y_path: str = "/home/ylzhang/Nano_HF/SunHaifeng/Model/Scaler/scaler_y.pkl"

    # Results save paths
    predictions_output_path: str = "/home/ylzhang/Nano_HF/OptiCPL/Results/PREDICTIONS_fabrication_params.csv"

    # Stage2 configuration - fixed seeds, not user-configurable
    stage2_optical_seed: int = 13  
    stage2_img_seed: int = 28      

    # Stage2 source directories (models loaded from here, saved to log_dir)
    stage2_img_model_src_dir: str = "/home/ylzhang/Nano_HF/OptiCPL/Checkpoints/Glum/IMG/GlumIMG"
    stage2_img_scaler_src_dir: str = "/home/ylzhang/Nano_HF/OptiCPL/Checkpoints/Glum/IMG/GlumIMG"
    stage2_optical_model_src_dir: str = "/home/ylzhang/Nano_HF/OptiCPL/Checkpoints/Glum/Optical/GlumOptical"
    stage2_optical_scaler_src_dir: str = "/home/ylzhang/Nano_HF/OptiCPL/Checkpoints/Glum/Optical/GlumOptical"

    # Glum data source path
    glum_data_src_path: str = "/home/ylzhang/Nano_HF/SunHaifeng/Data/glum/processed_merged_glum_15.xlsx"

    # Raw data pipeline configuration
    raw_data_path: str = "/home/ylzhang/Nano_HF/OptiCPL/Data/Data_complete_corrected_v7.xlsx"
    raw_pipeline_img_dir: str = "/home/ylzhang/Nano_HF/OptiCPL/Data/SEM_ALL"
    raw_pipeline_vae_model_path: str = "/home/ylzhang/Nano_HF/OptiCPL/Checkpoints/Pretrained/Spec/Spec_model.pth"
    raw_pipeline_img_model_path: str = "/home/ylzhang/Nano_HF/OptiCPL/Checkpoints/Pretrained/IMG/trained_resnet18_simclr2.pth"

    # Raw pipeline output paths (relative to log_dir, set during raw pipeline execution)
    raw_glum_output_path: Optional[str] = None
    raw_optical_output_path: Optional[str] = None
    raw_img_output_path: Optional[str] = None
    raw_combined_features_output_path: Optional[str] = None

    # Pipeline output paths (relative to log_dir, set during pipeline execution)
    stage2_features_output_path: Optional[str] = None
    pipeline_predictions_output_path: Optional[str] = None

    # Model architecture parameters
    hidden_dim1: int = 128
    hidden_dim2: int = 64
    hidden_dim3: int = 32
    hidden_dim4: int = 16
    dropout_rate: float = 0.1

    # Training hyperparameters
    learning_rate: float = 0.0005
    batch_size: int = 32
    num_epochs: int = 4000

    # Early stopping parameters
    patience: int = 10

    # Data split ratios
    test_size: float = 0.2  # 20% for test set
    val_size: float = 0.25  # 25% of remaining for validation (20% of total)

    # Target columns for prediction
    target_columns: List[str] = field(default_factory=list)

    # Data column names
    img_column: str = "IMG"
    glum_column: str = "glum"
    optical_column: str = "Optical"

    # Device (auto-initialized)
    device = None

    def __post_init__(self):
        """Initialize target columns after dataclass creation"""
        if not self.target_columns:
            self.target_columns = [
                'Deposition_angle', 'Deposition_speed', 'β', 'pitch',
                'n', 'CsBr', 'PbBr2', 'Soaking_time'
            ]

    def setup_training_paths(self, log_dir: str):
        """
        Update all save paths to use the training log directory.

        Args:
            log_dir: The training log directory path
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Update model paths
        self.model_save_dir = os.path.join(log_dir, "models")
        self.model_checkpoint_path = os.path.join(log_dir, "models", "stage1_best_model.pth")

        # Update scaler paths
        self.scaler_x_path = os.path.join(log_dir, "scalers", "scaler_x.pkl")
        self.scaler_y_path = os.path.join(log_dir, "scalers", "scaler_y.pkl")

        # Create a default predictions path in results subdirectory
        self.predictions_output_path = os.path.join(log_dir, "results", "predictions.csv")

    def setup_pipeline_paths(self, log_dir: str):
        """
        Set up paths for pipeline execution (stage2 + stage1)

        Args:
            log_dir: The pipeline log directory path
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Set stage2 features output path
        self.stage2_features_output_path = os.path.join(log_dir, "features", "stage2_features.csv")

        # Set stage1 model and scaler paths (like in training)
        self.model_save_dir = os.path.join(log_dir, "models")
        self.model_checkpoint_path = os.path.join(log_dir, "models", "stage1_best_model.pth")
        self.scaler_x_path = os.path.join(log_dir, "scalers", "scaler_x.pkl")
        self.scaler_y_path = os.path.join(log_dir, "scalers", "scaler_y.pkl")

        # Set predictions output path
        self.pipeline_predictions_output_path = os.path.join(log_dir, "results", "predictions.csv")

        # Ensure directories exist
        os.makedirs(os.path.dirname(self.stage2_features_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.pipeline_predictions_output_path), exist_ok=True)

        # Store log_dir for later use
        self.log_dir = log_dir

    def setup_raw_pipeline_paths(self, log_dir: str):
        """
        Set up paths for raw pipeline execution

        Args:
            log_dir: The raw pipeline log directory path
        """
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Set raw pipeline feature output paths
        features_dir = os.path.join(log_dir, "features")
        self.raw_glum_output_path = os.path.join(features_dir, "glum_parsed.csv")
        self.raw_optical_output_path = os.path.join(features_dir, "optical_features.csv")
        self.raw_img_output_path = os.path.join(features_dir, "img_features.csv")
        self.raw_combined_features_output_path = os.path.join(features_dir, "combined_features.csv")

        # Set stage1 model and scaler paths (like in training)
        self.model_save_dir = os.path.join(log_dir, "models")
        self.model_checkpoint_path = os.path.join(log_dir, "models", "stage1_best_model.pth")
        self.scaler_x_path = os.path.join(log_dir, "scalers", "scaler_x.pkl")
        self.scaler_y_path = os.path.join(log_dir, "scalers", "scaler_y.pkl")

        # Set predictions output path
        self.predictions_output_path = os.path.join(log_dir, "results", "predictions.csv")

        # Ensure directories exist
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.scaler_x_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.predictions_output_path), exist_ok=True)

        # Store log_dir for later use
        self.log_dir = log_dir

    def get_visualization_path(self, filename: str) -> str:
        """
        Get path for visualization files.

        Args:
            filename: Name of the visualization file

        Returns:
            Full path to the visualization file
        """
        if self.log_dir:
            return os.path.join(self.log_dir, "visualizations", filename)
        else:
            return os.path.join(self.model_save_dir, filename)

    def get_results_path(self, filename: str) -> str:
        """
        Get path for results files.

        Args:
            filename: Name of the results file

        Returns:
            Full path to the results file
        """
        if self.log_dir:
            return os.path.join(self.log_dir, "results", filename)
        else:
            return os.path.join(os.path.dirname(self.predictions_output_path), filename)
