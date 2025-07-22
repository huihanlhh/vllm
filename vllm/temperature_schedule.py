# SPDX-License-Identifier: Apache-2.0
"""Dynamic Temperature Scheduling for text generation."""
import copy
from typing import Annotated, Any, Optional, Union, Callable
import math
import torch
import torch.nn as nn
import torch.optim as optim

from vllm.logger import init_logger


logger = init_logger(__name__)


class FeatureEncoder:
    def __init__(self, entropy_len: int = 5):
        self.entropy_len = entropy_len

    def encode(self, features: dict) -> torch.Tensor:
        """
        Flatten structured features:
        - prev_temp: float
        - step_idx: int
        - prev_entropies: list[float] (padded/truncated)
        Returns a 1D float tensor.
        """
        vec = []

        vec.append(float(features.get("prev_temp", 1.0)))
        vec.append(float(features.get("step_idx", 0)))

        ent = features.get("prev_entropies", [])
        ent = (ent + [0.0] * self.entropy_len)[:self.entropy_len]
        vec.extend(ent)

        return torch.tensor(vec, dtype=torch.float32)


class SimpleTempNet(nn.Module):
    def __init__(self, input_dim: int = 7, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]; scaled to desired range
        )

    def forward(self, x):
        return self.net(x)


class TemperatureScheduler:
    def __init__(self, name, learned_model=None, learned_temp_amp=0.3, feature_encoder=None):
        self.name = name
        self.learned_temp_amp = learned_temp_amp

        if name == "learned":
            self.feature_encoder = feature_encoder or FeatureEncoder()
            input_dim = 2 + self.feature_encoder.entropy_len
            self.learned_model = learned_model or SimpleTempNet(input_dim=input_dim)
        else:
            self._temp_fn = self._get_temp_fn(name)

    def _get_temp_fn(self, name):
        if name == "constant":
            return self._constant
        elif name == "sinusoidal":
            return self._sinusoidal
        else:
            raise ValueError(f"Unknown temperature schedule: {name}")

    def set_temp(self, step_idx, init_temp, features=None, device='cpu', **kwargs):
        if self.name == "learned":
            return self._learned_temp(features=features, device=device)
        return self._temp_fn(step_idx, init_temp, **kwargs)

    def _constant(self, step_idx, init_temp, **kwargs):
        return init_temp

    def _sinusoidal(self, step_idx, init_temp, amp=0.3, period=160, **kwargs):
        phase = (2 * math.pi * step_idx) / period
        return init_temp + amp * math.sin(phase)

    def _learned_temp(self, features, device='cpu'):
        if features is None:
            raise ValueError("Must provide `features` for learned temperature.")

        x = self.feature_encoder.encode(features).unsqueeze(0).to(device)
        self.learned_model.to(device)
        self.learned_model.eval()
        with torch.no_grad():
            norm_temp = self.learned_model(x).squeeze()
            low, high = self.learned_temp_range
            return low + (high - low) * norm_temp.item()

    def train_learned(self, train_feature_dicts, target_temps, lr=1e-3, epochs=100, device="cpu"):
        """Train the learned model on structured features."""
        x_list = [self.feature_encoder.encode(f) for f in train_feature_dicts]
        x = torch.stack(x_list).to(device)
        y = torch.tensor(target_temps, dtype=torch.float32).to(device)

        low, high = self.learned_temp_range
        y_norm = (y - low) / (high - low)

        self.learned_model.to(device)
        self.learned_model.train()

        optimizer = optim.Adam(self.learned_model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = self.learned_model(x).squeeze()
            loss = loss_fn(pred, y_norm)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"[Epoch {epoch+1}/{epochs}] Loss: {loss.item():.4f}")

    def clone(self) -> "TemperatureScheduler":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return f"TemperatureScheduler(name='{self.name}')"