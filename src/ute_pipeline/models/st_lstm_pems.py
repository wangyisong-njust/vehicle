"""ST-LSTM models for PeMS sensor-network long-horizon traffic forecasting.

Contains both regression variants (kept for reference) and the classification
variants used in the current long-horizon 4-class state forecasting experiment.
Classification head mirrors the OBB-ST-LSTM classification head (CrossEntropy,
per-window 4-class output), transferred to the PeMS multi-sensor setting.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .obb_st_lstm import DisturbanceGatedLSTM


class STLSTMRegressor(nn.Module):
    """ST-LSTM for sensor-network long-horizon regression.

    Architecture transfer of OBB-ST-LSTM with two critical adaptations for
    multi-sensor regression on PeMS:
      (1) The 1D-CNN preserves the sensor axis (no GAP) so per-sensor identity
          is retained; spatial mixing happens through small-kernel conv.
      (2) The model predicts a *residual on top of the last observed value*
          (persistence prior), which is standard in long-horizon traffic flow
          prediction and makes the model competitive with the strong
          Persistence baseline at short horizons.

    The combination "spatial-CNN + temporal-LSTM + persistence prior" mirrors
    the OBB-ST-LSTM idea of fusing spatial and temporal information into a
    single end-to-end model, with no second model and no late-stage weighted
    fusion.
    """

    def __init__(
        self,
        in_channels: int,
        sensors: int,
        hidden_size: int = 64,
        conv_channels: tuple[int, int] = (4, 4),
        dropout: float = 0.1,
    ):
        super().__init__()
        c1, c2 = conv_channels
        # 1D conv mixes neighbouring sensors but keeps the sensor axis intact.
        self.spatial_encoder = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.spatial_out = c2
        # Per-step feature vector = c2 channels x sensors (flat). Compact LSTM
        # then summarises temporal evolution; output head predicts a per-sensor
        # residual on top of the persistence prior.
        self.lstm = nn.LSTM(c2 * sensors, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, sensors),
        )
        self.sensors = int(sensors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, 1, N)
        b, t, c, h, w = x.shape
        assert h == 1, "STLSTMRegressor expects sensor input with H=1"
        # Persistence prior: use the last observation of channel 0
        last_obs = x[:, -1, 0, 0, :]  # (B, N)
        flat = x.reshape(b * t, c, w)
        z = self.spatial_encoder(flat)  # (B*T, c2, N)
        z = z.reshape(b, t, self.spatial_out * w)  # (B, T, c2*N) -- flat per-step
        out, _ = self.lstm(z)  # (B, T, hidden)
        delta = self.head(out[:, -1, :])  # head returns (B, N) when last linear is Linear(hidden, N)
        return last_obs + delta


class LSTMRegressor(nn.Module):
    """Pure scalar LSTM baseline that flattens the sensor axis into the input."""

    def __init__(self, input_dim: int, sensors: int, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, sensors),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class GRURegressor(nn.Module):
    def __init__(self, input_dim: int, sensors: int, hidden_size: int = 64, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, sensors),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class STLSTMClassifier(nn.Module):
    """ST-LSTM for per-sensor 4-class traffic state classification on PeMS data.

    Architecture mirrors OBB-ST-LSTM: 1D-CNN spatial encoder → LSTM → per-sensor
    class logits. Head outputs (B, N, n_classes) so each sensor independently
    predicts its future traffic state.
    """

    def __init__(
        self,
        in_channels: int,
        sensors: int,
        hidden_size: int = 64,
        conv_channels: tuple[int, ...] = (8, 16),
        n_classes: int = 4,
        dropout: float = 0.1,
        bidirectional: bool = True,
        use_disturbance_gate: bool = True,
        disturbance_dim: int = 1,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_c = in_channels
        for c in conv_channels:
            layers.append(nn.Conv1d(prev_c, c, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            prev_c = c
        layers.append(nn.Dropout(dropout))
        self.spatial_encoder = nn.Sequential(*layers)
        c_last = conv_channels[-1]
        self.spatial_out = c_last
        self.bidirectional = bool(bidirectional)
        self.use_disturbance_gate = bool(use_disturbance_gate)
        self.disturbance_dim = int(disturbance_dim)
        if self.use_disturbance_gate:
            self.lstm = DisturbanceGatedLSTM(
                c_last * sensors,
                hidden_size,
                disturbance_dim=self.disturbance_dim,
                bidirectional=self.bidirectional,
            )
        else:
            self.lstm = nn.LSTM(c_last * sensors, hidden_size, batch_first=True, bidirectional=self.bidirectional)
        lstm_out = hidden_size * (2 if self.bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_out, sensors * n_classes),
        )
        self.sensors = int(sensors)
        self.n_classes = int(n_classes)

    def forward(self, x: torch.Tensor, disturbance: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T, C, 1, N)
        b, t, c, h, w = x.shape
        flat = x.reshape(b * t, c, w)
        z = self.spatial_encoder(flat)  # (B*T, c2, N)
        z = z.reshape(b, t, self.spatial_out * w)
        if self.use_disturbance_gate:
            out, _ = self.lstm(z, disturbance=disturbance)
        else:
            out, _ = self.lstm(z)
        if self.bidirectional:
            hs = out.shape[-1] // 2
            feat = torch.cat([out[:, -1, :hs], out[:, 0, hs:]], dim=-1)
        else:
            feat = out[:, -1, :]
        return self.head(feat).reshape(b, self.sensors, self.n_classes)  # (B, N, 4)


class LSTMClassifier(nn.Module):
    """Scalar LSTM baseline for per-sensor traffic state classification."""

    def __init__(self, input_dim: int, sensors: int, hidden_size: int = 64, n_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, sensors * n_classes))
        self.sensors = int(sensors)
        self.n_classes = int(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).reshape(x.shape[0], self.sensors, self.n_classes)


class GRUClassifier(nn.Module):
    """GRU baseline for per-sensor traffic state classification."""

    def __init__(self, input_dim: int, sensors: int, hidden_size: int = 64, n_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, sensors * n_classes))
        self.sensors = int(sensors)
        self.n_classes = int(n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).reshape(x.shape[0], self.sensors, self.n_classes)


def fit_classifier(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    seed: int,
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    device: str | None = None,
    loss_fn: nn.Module | None = None,
    train_disturbance: torch.Tensor | None = None,
    test_disturbance: torch.Tensor | None = None,
) -> np.ndarray:
    """Train a classifier; return softmax probabilities (B, N, 4).

    `loss_fn` defaults to CrossEntropyLoss. Pass FocalLoss(gamma, weight) for
    class-imbalance-aware training. Loss is computed on flattened (B*N, C)
    logits / (B*N,) labels so it works with both CrossEntropy and FocalLoss.
    """
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    train_x_d = train_x.to(device)
    train_y_d = train_y.to(device)  # (B, N) long
    test_x_d = test_x.to(device)
    train_disturbance_d = train_disturbance.to(device) if train_disturbance is not None else None
    test_disturbance_d = test_disturbance.to(device) if test_disturbance is not None else None
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    loss_fn = loss_fn.to(device)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(train_x_d.shape[0], device=device)
        for start in range(0, train_x_d.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            if train_disturbance_d is not None:
                logits = model(train_x_d[idx], disturbance=train_disturbance_d[idx])
            else:
                logits = model(train_x_d[idx])  # (B, N, 4)
            n_classes = logits.shape[-1]
            flat_logits = logits.reshape(-1, n_classes)
            flat_target = train_y_d[idx].reshape(-1)
            loss = loss_fn(flat_logits, flat_target)
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
        scheduler.step()
    model.eval()
    out_chunks: list[torch.Tensor] = []
    eval_batch = max(1, batch_size)
    with torch.no_grad():
        for start in range(0, test_x_d.shape[0], eval_batch):
            if test_disturbance_d is not None:
                chunk = model(
                    test_x_d[start : start + eval_batch],
                    disturbance=test_disturbance_d[start : start + eval_batch],
                )
            else:
                chunk = model(test_x_d[start : start + eval_batch])
            out_chunks.append(torch.softmax(chunk, dim=-1).cpu())
    probs = torch.cat(out_chunks, dim=0)
    return probs.numpy()  # (B, N, 4)


def fit_regressor(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    seed: int,
    epochs: int = 25,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    device: str | None = None,
) -> np.ndarray:
    torch.manual_seed(seed)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    train_x_d = train_x.to(device)
    train_y_d = train_y.to(device)
    test_x_d = test_x.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.SmoothL1Loss()
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(train_x_d.shape[0], device=device)
        for start in range(0, train_x_d.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            pred = model(train_x_d[idx])
            loss = loss_fn(pred, train_y_d[idx])
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
    model.eval()
    with torch.no_grad():
        out = model(test_x_d).cpu().numpy()
    return out
