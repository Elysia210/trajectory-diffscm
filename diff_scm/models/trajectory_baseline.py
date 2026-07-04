"""
Minimal GRU baseline for pair-level trajectory collision classification.
"""

import torch
from torch import nn


class TrajectoryGRUBaseline(nn.Module):
    """
    Encode a [B, T, F] trajectory sequence and predict one collision logit.
    """

    def __init__(
        self,
        input_dim: int = 23,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        # Input standardization stats. Identity by default (mean 0 / std 1) so the
        # module stays backward compatible and is a no-op until populated at train
        # time. Stored as buffers so they are saved in the checkpoint and reused
        # automatically wherever the classifier is loaded (incl. the sampler).
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_std", torch.ones(input_dim))
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        # Standardize per-feature before the recurrent encoder. World-coordinate
        # positions can reach thousands while angular features are O(1); without
        # this the GRU saturates and the classifier fails to learn.
        trajectory = (trajectory - self.feature_mean) / self.feature_std
        _, hidden = self.encoder(trajectory)
        if self.encoder.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]
        return self.classifier(hidden)
