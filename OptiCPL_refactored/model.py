"""
Model architecture module for Stage 1 Model
Neural network definition with Layer Normalization and Dropout
"""

import torch
import torch.nn as nn
from typing import Optional


class NeuralNet(nn.Module):
    """
    Neural network for predicting fabrication parameters from material features.

    Architecture:
    - 5 fully connected layers with decreasing dimensions
    - Layer Normalization after each hidden layer
    - ReLU activation functions
    - Dropout for regularization
    - Direct output (no activation on final layer)

    The network is designed to predict 8 fabrication parameters from
    combined image, glum, and optical features.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        hidden_dim3: int = 32,
        hidden_dim4: int = 16,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the neural network.

        Args:
            input_size: Number of input features
            output_size: Number of output targets
            hidden_dim1: Size of first hidden layer
            hidden_dim2: Size of second hidden layer
            hidden_dim3: Size of third hidden layer
            hidden_dim4: Size of fourth hidden layer
            dropout_rate: Dropout probability for regularization
        """
        super(NeuralNet, self).__init__()

        # Store dimensions
        self.input_size = input_size
        self.output_size = output_size

        # Fully connected layers
        self.fc1 = nn.Linear(input_size, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.fc5 = nn.Linear(hidden_dim4, output_size)

        # Layer normalization layers
        self.ln1 = nn.LayerNorm(hidden_dim1)
        self.ln2 = nn.LayerNorm(hidden_dim2)
        self.ln3 = nn.LayerNorm(hidden_dim3)
        self.ln4 = nn.LayerNorm(hidden_dim4)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate)

        # Activation function
        self.relu = nn.ReLU()

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_size)

        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Layer 1: FC -> LayerNorm -> ReLU -> Dropout
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 2: FC -> LayerNorm -> ReLU -> Dropout
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 3: FC -> LayerNorm -> ReLU -> Dropout
        x = self.fc3(x)
        x = self.ln3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 4: FC -> LayerNorm -> ReLU -> Dropout
        x = self.fc4(x)
        x = self.ln4(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output layer: FC only (no activation)
        x = self.fc5(x)

        return x

    def get_model_info(self) -> dict:
        """
        Get model architecture information.

        Returns:
            Dictionary containing model information
        """
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_layers": [self.fc1.out_features, self.fc2.out_features,
                            self.fc3.out_features, self.fc4.out_features],
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def create_model(input_size: int, output_size: int, config) -> NeuralNet:
    """
    Create a NeuralNet model from config.

    Args:
        input_size: Number of input features
        output_size: Number of output targets
        config: Configuration object

    Returns:
        Initialized NeuralNet model
    """
    return NeuralNet(
        input_size=input_size,
        output_size=output_size,
        hidden_dim1=config.hidden_dim1,
        hidden_dim2=config.hidden_dim2,
        hidden_dim3=config.hidden_dim3,
        hidden_dim4=config.hidden_dim4,
        dropout_rate=config.dropout_rate
    )
