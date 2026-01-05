"""
Stage 2 Models for OptiCPL
Handles models that predict features (IMG and Optical) from glum data
"""

import torch
import torch.nn as nn
import pickle
import re
from pathlib import Path
from typing import Tuple, Optional
import numpy as np


class GlumToOpticalModel(nn.Module):
    """
    Model to predict Optical features (32-dim) from glum data (40-dim)
    """

    def __init__(self, input_dim: int = 40, optical_dim: int = 32):
        super(GlumToOpticalModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3)
        )
        self.optical_head = nn.Linear(128, optical_dim)

    def forward(self, x):
        shared_output = self.shared(x)
        optical_output = self.optical_head(shared_output)
        return optical_output


class GlumToIMGModel(nn.Module):
    """
    Model to predict IMG features (128-dim) from glum data (40-dim)
    """

    def __init__(self, input_dim: int = 40, img_dim: int = 128):
        super(GlumToIMGModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4)
        )
        self.optical_head = nn.Linear(256, img_dim)

    def forward(self, x):
        shared_output = self.shared(x)
        optical_output = self.optical_head(shared_output)
        return optical_output


class Stage2ModelManager:
    """
    Manager for loading and using Stage 2 models
    """

    # Fixed seeds as per requirements
    OPTICAL_SEED = 13
    IMG_SEED = 28

    def __init__(self,
                 img_model_src_dir: str,
                 img_scaler_src_dir: str,
                 optical_model_src_dir: str,
                 optical_scaler_src_dir: str):
        """
        Initialize the model manager

        Args:
            img_model_src_dir: Directory containing IMG models
            img_scaler_src_dir: Directory containing IMG scalers
            optical_model_src_dir: Directory containing Optical models
            optical_scaler_src_dir: Directory containing Optical scalers
        """
        self.img_model_src_dir = Path(img_model_src_dir)
        self.img_scaler_src_dir = Path(img_scaler_src_dir)
        self.optical_model_src_dir = Path(optical_model_src_dir)
        self.optical_scaler_src_dir = Path(optical_scaler_src_dir)

    def load_optical_model(self, device: str = 'cpu') -> Tuple[GlumToOpticalModel, object, object]:
        """
        Load Optical model and scalers for fixed seed 13

        Args:
            device: Device to load model on

        Returns:
            Tuple of (model, scaler_X, scaler_Y)
        """
        seed = self.OPTICAL_SEED

        # Find model file - try pattern first, then fallback to model.pth
        pattern = f"*seed{seed}_r2_*.pth"
        model_files = list(self.optical_model_src_dir.glob(pattern))

        if model_files:
            # Use the pattern-matched file
            model_file = model_files[0]
            model_filename = model_file.name

            # Extract identifier
            match = re.search(rf"seed{seed}_r2_([\d_]+)", model_filename)
            if not match:
                raise ValueError(f"Cannot extract identifier from filename: {model_filename}")

            r2_identifier = f"seed{seed}_r2_{match.group(1)}"

            # Load scalers with identifier
            scaler_X_file = self.optical_scaler_src_dir / f"scaler_X_{r2_identifier}.pkl"
            scaler_Y_file = self.optical_scaler_src_dir / f"scaler_Y_{r2_identifier}.pkl"
        else:
            # Fallback to model.pth
            model_file = self.optical_model_src_dir / "model.pth"
            if not model_file.exists():
                raise FileNotFoundError(
                    f"No Optical model found for seed {seed} in {self.optical_model_src_dir}. "
                    f"Tried pattern '{pattern}' and 'model.pth'."
                )

            # Fallback to simple scaler names
            scaler_X_file = self.optical_scaler_src_dir / "scaler_X.pkl"
            scaler_Y_file = self.optical_scaler_src_dir / "scaler_Y.pkl"

        # Check scaler files exist
        if not scaler_X_file.exists():
            raise FileNotFoundError(f"Scaler X not found: {scaler_X_file}")
        if not scaler_Y_file.exists():
            raise FileNotFoundError(f"Scaler Y not found: {scaler_Y_file}")

        # Load scalers
        with open(scaler_X_file, "rb") as f:
            scaler_X = pickle.load(f)
        with open(scaler_Y_file, "rb") as f:
            scaler_Y = pickle.load(f)

        # Initialize and load model
        model = GlumToOpticalModel(input_dim=40, optical_dim=32)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.to(device)
        model.eval()

        return model, scaler_X, scaler_Y

    def load_img_model(self, device: str = 'cpu') -> Tuple[GlumToIMGModel, object, object]:
        """
        Load IMG model and scalers for fixed seed 28

        Args:
            device: Device to load model on

        Returns:
            Tuple of (model, scaler_X, scaler_Y)
        """
        seed = self.IMG_SEED

        # Find model file - try pattern first, then fallback to model.pth
        pattern = f"*seed{seed}_r2_*.pth"
        model_files = list(self.img_model_src_dir.glob(pattern))

        if model_files:
            # Use the pattern-matched file
            model_file = model_files[0]
            model_filename = model_file.name

            # Extract identifier
            match = re.search(rf"seed{seed}_r2_([\d_]+)", model_filename)
            if not match:
                raise ValueError(f"Cannot extract identifier from filename: {model_filename}")

            r2_identifier = f"seed{seed}_r2_{match.group(1)}"

            # Load scalers with identifier
            scaler_X_file = self.img_scaler_src_dir / f"scaler_X_{r2_identifier}.pkl"
            scaler_Y_file = self.img_scaler_src_dir / f"scaler_Y_{r2_identifier}.pkl"
        else:
            # Fallback to model.pth
            model_file = self.img_model_src_dir / "model.pth"
            if not model_file.exists():
                raise FileNotFoundError(
                    f"No IMG model found for seed {seed} in {self.img_model_src_dir}. "
                    f"Tried pattern '{pattern}' and 'model.pth'."
                )

            # Fallback to simple scaler names
            scaler_X_file = self.img_scaler_src_dir / "scaler_X.pkl"
            scaler_Y_file = self.img_scaler_src_dir / "scaler_Y.pkl"

        # Check scaler files exist
        if not scaler_X_file.exists():
            raise FileNotFoundError(f"Scaler X not found: {scaler_X_file}")
        if not scaler_Y_file.exists():
            raise FileNotFoundError(f"Scaler Y not found: {scaler_Y_file}")

        # Load scalers
        with open(scaler_X_file, "rb") as f:
            scaler_X = pickle.load(f)
        with open(scaler_Y_file, "rb") as f:
            scaler_Y = pickle.load(f)

        # Initialize and load model
        model = GlumToIMGModel(input_dim=40, img_dim=128)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.to(device)
        model.eval()

        return model, scaler_X, scaler_Y
