"""
Stage 1 Model - Predicts fabrication parameters from material features

A professional PyTorch implementation for predicting 8 fabrication parameters
from combined image, glum, and optical material features.
"""

__version__ = "1.0.0"
__author__ = "Research Team"

from .config import Config
from .model import NeuralNet, create_model
from .data_utils import (
    load_training_data, load_inference_data,
    prepare_features, prepare_labels, create_dataloaders,
    load_scalers, save_scalers
)
from .trainer import (
    train_model, evaluate_model,
    save_model_checkpoint, load_model_checkpoint
)
from .inference import (
    make_predictions, run_inference_from_dataframe,
    run_inference_from_file, save_predictions
)
from .stage2_models import Stage2ModelManager
from .stage2_inference import (
    generate_stage2_features, parse_glum_data,
    predict_optical, predict_img
)
from .pipeline import run_end_to_end_pipeline
from .visualization import (
    plot_training_curves, plot_prediction_comparison,
    plot_residuals, plot_error_distribution
)

__all__ = [
    # Config
    'Config',

    # Model
    'NeuralNet',
    'create_model',

    # Data utilities
    'load_training_data',
    'load_inference_data',
    'prepare_features',
    'prepare_labels',
    'create_dataloaders',
    'load_scalers',
    'save_scalers',

    # Trainer
    'train_model',
    'evaluate_model',
    'save_model_checkpoint',
    'load_model_checkpoint',

    # Inference
    'make_predictions',
    'run_inference_from_dataframe',
    'run_inference_from_file',
    'save_predictions',

    # Stage2 models and inference
    'Stage2ModelManager',
    'generate_stage2_features',
    'parse_glum_data',
    'predict_optical',
    'predict_img',

    # Pipeline
    'run_end_to_end_pipeline',

    # Visualization
    'plot_training_curves',
    'plot_prediction_comparison',
    'plot_residuals',
    'plot_error_distribution',
]
