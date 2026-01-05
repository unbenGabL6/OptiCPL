"""
VAE Model for Optical Feature Extraction
Handles loading and inference of the VAE model for generating Optical features from spectral data
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Optional
import ast


class MultiInputVAE(nn.Module):
    """
    Multi-Input Variational Autoencoder for processing 5 types of spectral data
    Input: PL(40), EX_sample(200), CD_sample(200), CD_template(200), EX_template(200)
    Output: 32-dim latent features
    """

    def __init__(self, dims, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoders for each input type
        self.pl_encoder = self._build_encoder(dims[0], latent_dim)
        self.ex_sample_encoder = self._build_encoder(dims[1], latent_dim)
        self.cd_sample_encoder = self._build_encoder(dims[2], latent_dim)
        self.cd_template_encoder = self._build_encoder(dims[3], latent_dim)
        self.ex_template_encoder = self._build_encoder(dims[4], latent_dim)

        # Fusion layer
        self.fc_fusion = nn.Linear(latent_dim * 5, latent_dim)

        # Independent decoders
        self.pl_decoder = self._build_decoder(latent_dim, dims[0])
        self.ex_sample_decoder = self._build_decoder(latent_dim, dims[1])
        self.cd_sample_decoder = self._build_decoder(latent_dim, dims[2])
        self.cd_template_decoder = self._build_decoder(latent_dim, dims[3])
        self.ex_template_decoder = self._build_decoder(latent_dim, dims[4])

    def _build_encoder(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def _build_decoder(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, out_dim),
            nn.Sigmoid()
        )

    def forward(self, pl, ex_sample, cd_sample, cd_template, ex_template):
        # Encode each input
        pl_feat = self.pl_encoder(pl)
        ex_sample_feat = self.ex_sample_encoder(ex_sample)
        cd_sample_feat = self.cd_sample_encoder(cd_sample)
        cd_template_feat = self.cd_template_encoder(cd_template)
        ex_template_feat = self.ex_template_encoder(ex_template)

        # Fuse features
        fused = torch.cat([pl_feat, ex_sample_feat, cd_sample_feat,
                           cd_template_feat, ex_template_feat], dim=1)
        z = torch.relu(self.fc_fusion(fused))

        # Decode (for training, but we only need encoding for inference)
        pl_recon = self.pl_decoder(z)
        ex_sample_recon = self.ex_sample_decoder(z)
        cd_sample_recon = self.cd_sample_decoder(z)
        cd_template_recon = self.cd_template_decoder(z)
        ex_template_recon = self.ex_template_decoder(z)

        return [pl_recon, ex_sample_recon, cd_sample_recon,
                cd_template_recon, ex_template_recon], z

    def encode(self, pl, ex_sample, cd_sample, cd_template, ex_template):
        """Encode inputs to latent space (inference only)"""
        pl_feat = self.pl_encoder(pl)
        ex_sample_feat = self.ex_sample_encoder(ex_sample)
        cd_sample_feat = self.cd_sample_encoder(cd_sample)
        cd_template_feat = self.cd_template_encoder(cd_template)
        ex_template_feat = self.ex_template_encoder(ex_template)

        fused = torch.cat([pl_feat, ex_sample_feat, cd_sample_feat,
                           cd_template_feat, ex_template_feat], dim=1)
        z = torch.relu(self.fc_fusion(fused))
        return z


def load_vae_model(checkpoint_path: str, device: str = 'cpu') -> MultiInputVAE:
    """
    Load VAE model from checkpoint

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Device to load model on ('cpu' or 'cuda')

    Returns:
        Loaded VAE model in eval mode
    """
    # Model dimensions
    dims = [40, 200, 200, 200, 200]  # PL, EX_sample, CD_sample, CD_template, EX_template
    latent_dim = 32

    # Initialize model
    model = MultiInputVAE(dims, latent_dim)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    # Move to device and set eval mode
    model.to(device)
    model.eval()

    return model


def min_max_scale(array_2d, min_val=None, max_val=None):
    """
    Min-max normalization to [0, 1]

    Args:
        array_2d: Input array of shape [num_samples, spectrum_length]
        min_val: Minimum value (if None, computed from data)
        max_val: Maximum value (if None, computed from data)

    Returns:
        Normalized array
    """
    if min_val is None:
        min_val = np.min(array_2d)
    if max_val is None:
        max_val = np.max(array_2d)

    # Avoid division by zero
    if max_val == min_val:
        return array_2d - min_val

    return (array_2d - min_val) / (max_val - min_val)


def preprocess_spectral_data(df: pd.DataFrame, device: str = 'cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preprocess spectral data from DataFrame

    Args:
        df: DataFrame containing spectral columns
        device: Device to move tensors to

    Returns:
        Tuple of (pl, ex_sample, cd_sample, cd_template, ex_template) tensors
    """

    def safe_eval(val):
        """Safely evaluate string to array"""
        try:
            if isinstance(val, str):
                return ast.literal_eval(val)
            return val
        except (ValueError, SyntaxError):
            return None

    # Parse spectral columns
    spectral_data = {}
    for col in ['PL', 'EX_sample', 'CD_sample', 'CD_template', 'EX_template']:
        parsed_list = df[col].apply(safe_eval).dropna().tolist()
        spectral_data[col] = np.array(parsed_list, dtype=np.float32)

    # Preprocess each spectral type
    processed_data = {}
    for col, data in spectral_data.items():
        # CD samples need special scaling
        if 'CD' in col:
            data = data * 1e-3  # Divide by 1000

        # Min-max normalization
        data_normalized = min_max_scale(data)
        processed_data[col] = data_normalized

    # Convert to tensors
    pl = torch.tensor(processed_data['PL'], dtype=torch.float32).to(device)
    ex_sample = torch.tensor(processed_data['EX_sample'], dtype=torch.float32).to(device)
    cd_sample = torch.tensor(processed_data['CD_sample'], dtype=torch.float32).to(device)
    cd_template = torch.tensor(processed_data['CD_template'], dtype=torch.float32).to(device)
    ex_template = torch.tensor(processed_data['EX_template'], dtype=torch.float32).to(device)

    return pl, ex_sample, cd_sample, cd_template, ex_template


def extract_optical_features(vae_model: MultiInputVAE, df: pd.DataFrame, device: str = 'cpu',
                             batch_size: int = 32) -> np.ndarray:
    """
    Extract Optical features using VAE model

    Args:
        vae_model: Loaded VAE model
        df: DataFrame containing spectral data
        device: Device to run inference on
        batch_size: Batch size for inference

    Returns:
        Array of Optical features [N, 32]
    """
    # Preprocess data
    pl, ex_sample, cd_sample, cd_template, ex_template = preprocess_spectral_data(df, device)

    # Create dataset and dataloader
    dataset = TensorDataset(pl, ex_sample, cd_sample, cd_template, ex_template)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Extract features
    all_features = []
    vae_model.eval()

    with torch.no_grad():
        for batch in dataloader:
            batch_pl, batch_ex, batch_cd_s, batch_cd_t, batch_ex_t = [x.to(device) for x in batch]

            # Extract latent features
            features = vae_model.encode(batch_pl, batch_ex, batch_cd_s, batch_cd_t, batch_ex_t)
            all_features.append(features.cpu())

    # Concatenate all features
    all_features = torch.cat(all_features, dim=0).numpy()

    return all_features
