"""
PyTorch Deep Recurrent Progression Architectures (LSTM & GRU).
Provides modular model contracts for future prospective longitudinal datasets.
DOES NOT FABRICATE PRE-TRAINED WEIGHTS FOR STATIC DATA.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

class CoconutProgressionLSTM(nn.Module):
    """
    Long Short-Term Memory network for palm disease severity sequence modeling.
    Input shape: (batch_size, seq_len, input_dim)
    Output shape: (batch_size, 1) in range [0.0, 1.0]
    """
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        # Pool last sequence step representation
        last_step = out[:, -1, :]
        severity_pred = self.regressor(last_step)
        return severity_pred

class CoconutProgressionGRU(nn.Module):
    """
    Gated Recurrent Unit network for palm disease severity sequence modeling.
    Input shape: (batch_size, seq_len, input_dim)
    Output shape: (batch_size, 1) in range [0.0, 1.0]
    """
    def __init__(self, input_dim: int = 10, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last_step = out[:, -1, :]
        severity_pred = self.regressor(last_step)
        return severity_pred
