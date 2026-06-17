from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


class DisturbanceGatedLSTM(nn.Module):
    """Single-layer LSTM with an extra disturbance gate on candidate memory writes."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        disturbance_dim: int = 1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.disturbance_dim = int(disturbance_dim)
        self.bidirectional = bool(bidirectional)
        self.forward_cell = DisturbanceGatedLSTMCell(input_size, hidden_size, disturbance_dim)
        self.reverse_cell = (
            DisturbanceGatedLSTMCell(input_size, hidden_size, disturbance_dim)
            if self.bidirectional
            else None
        )

    @property
    def output_size(self) -> int:
        return self.hidden_size * (2 if self.bidirectional else 1)

    def _run_direction(
        self,
        cell: "DisturbanceGatedLSTMCell",
        x: torch.Tensor,
        disturbance: torch.Tensor,
        reverse: bool,
    ) -> torch.Tensor:
        b, t, _ = x.shape
        h = x.new_zeros(b, self.hidden_size)
        c = x.new_zeros(b, self.hidden_size)
        outputs = []
        indices = range(t - 1, -1, -1) if reverse else range(t)
        for step in indices:
            h, c = cell(x[:, step, :], disturbance[:, step, :], h, c)
            outputs.append(h)
        if reverse:
            outputs.reverse()
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor, disturbance: torch.Tensor | None = None) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if disturbance is None:
            disturbance = x.new_zeros(x.shape[0], x.shape[1], self.disturbance_dim)
        if disturbance.ndim == 2:
            disturbance = disturbance.unsqueeze(-1)
        if disturbance.shape[:2] != x.shape[:2]:
            raise ValueError("disturbance must have shape (B, T, D) aligned with x")
        if disturbance.shape[-1] != self.disturbance_dim:
            raise ValueError(f"Expected disturbance_dim={self.disturbance_dim}, got {disturbance.shape[-1]}")

        out_f = self._run_direction(self.forward_cell, x, disturbance, reverse=False)
        last_h = [out_f[:, -1, :]]
        last_c = [self.forward_cell.last_c]
        if self.reverse_cell is not None:
            out_r = self._run_direction(self.reverse_cell, x, disturbance, reverse=True)
            out = torch.cat([out_f, out_r], dim=-1)
            last_h.append(out_r[:, 0, :])
            last_c.append(self.reverse_cell.last_c)
        else:
            out = out_f
        return out, (torch.stack(last_h, dim=0), torch.stack(last_c, dim=0))


class DisturbanceGatedLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, disturbance_dim: int = 1):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.xh = nn.Linear(int(input_size) + self.hidden_size, 4 * self.hidden_size)
        self.disturbance_gate = nn.Linear(int(disturbance_dim), self.hidden_size)
        nn.init.constant_(self.disturbance_gate.bias, -2.0)
        self.last_c: torch.Tensor | None = None

    def forward(
        self,
        x_t: torch.Tensor,
        d_t: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        xh = torch.cat([x_t, h_prev], dim=-1)
        i, f, g, o = self.xh(xh).chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        r = torch.sigmoid(self.disturbance_gate(d_t))
        c = f * c_prev + i * (1.0 + r) * g
        h = o * torch.tanh(c)
        self.last_c = c
        return h, c


class GTSEPDL(nn.Module):
    """OBB-aware Spatio-Temporal LSTM.

    Input:
        x_tensor: (B, T, C, H, W) per-frame OBB grid tensor
            ch0: OBB cell occupancy (HF-GO)
            ch1: HBB cell occupancy
            ch2: cell-weighted sin(theta)
            ch3: cell-weighted cos(theta)
        x_scalar (optional): (B, T, F) per-frame handcrafted traffic descriptors
            (V/D/F features). When provided, they are concatenated to the CNN
            spatial embedding before being fed into the LSTM. This still forms
            a single end-to-end model (no separate "second model" / no
            late-stage weighted fusion).
    Output: (B, num_classes) logits.

    Frontend = 2-layer Conv2d encoder with GAP -> per-frame spatial embedding,
               optionally concatenated with scalar context features.
    Backbone = single-layer LSTM, last-step output -> linear head.
    """

    def __init__(
        self,
        in_channels: int = 4,
        conv_channels: tuple[int, ...] = (16, 32, 32),
        hidden_size: int = 64,
        num_classes: int = 4,
        dropout: float = 0.25,
        scalar_dim: int = 0,
        bidirectional: bool = True,
        lstm_layers: int = 1,
        use_disturbance_gate: bool = True,
        disturbance_dim: int = 1,
        use_persistence_prior: bool = False,
        persistence_alpha_init: float = 1.0,
        channel_dropout_p: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev_c = in_channels
        for c in conv_channels:
            layers.append(nn.Conv2d(prev_c, c, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            prev_c = c
        layers.append(nn.Dropout2d(dropout))
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.spatial_encoder = nn.Sequential(*layers)
        spatial_out = conv_channels[-1]
        self.scalar_dim = int(scalar_dim)
        self.embed_dim = spatial_out + self.scalar_dim
        self.use_disturbance_gate = bool(use_disturbance_gate)
        self.disturbance_dim = int(disturbance_dim)
        if self.use_disturbance_gate:
            self.lstm = DisturbanceGatedLSTM(
                self.embed_dim,
                hidden_size,
                disturbance_dim=self.disturbance_dim,
                bidirectional=bool(bidirectional),
            )
        else:
            self.lstm = nn.LSTM(
                self.embed_dim,
                hidden_size,
                num_layers=int(lstm_layers),
                batch_first=True,
                bidirectional=bool(bidirectional),
                dropout=float(dropout) if int(lstm_layers) > 1 else 0.0,
            )
        head_in = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, num_classes),
        )
        self.bidirectional = bool(bidirectional)
        self.hidden_size = int(hidden_size)
        self.num_classes = int(num_classes)
        self.use_persistence_prior = bool(use_persistence_prior)
        self.channel_dropout_p = float(channel_dropout_p)
        if self.use_persistence_prior:
            # Learnable bias toward the current observed state. At long horizons
            # (5s/8s ahead) the strongest prior is "stay at current state"; we
            # let the model learn how strongly to inject the current-state
            # one-hot into the logits.
            self.persistence_alpha = nn.Parameter(torch.tensor(float(persistence_alpha_init)))

    def forward(
        self,
        x: torch.Tensor,
        x_scalar: torch.Tensor | None = None,
        disturbance: torch.Tensor | None = None,
        current_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b, t, c, h, w = x.shape
        if self.training and self.channel_dropout_p > 0:
            # Channel-wise dropout for data augmentation: randomly zero out
            # entire spatial channels per sample with probability p.
            mask = (torch.rand(b, 1, c, 1, 1, device=x.device) > self.channel_dropout_p).float()
            x = x * mask
        flat = x.reshape(b * t, c, h, w)
        z_spatial = self.spatial_encoder(flat).reshape(b, t, -1)
        if self.scalar_dim > 0:
            if x_scalar is None:
                raise ValueError("scalar_dim>0 but x_scalar is None")
            z = torch.cat([z_spatial, x_scalar], dim=-1)
        else:
            z = z_spatial
        if self.use_disturbance_gate:
            out, _ = self.lstm(z, disturbance=disturbance)
        else:
            out, _ = self.lstm(z)
        logits = self.head(out[:, -1, :])
        if self.use_persistence_prior:
            if current_state is None:
                raise ValueError("use_persistence_prior=True but current_state is None")
            prior = nn.functional.one_hot(current_state.long(), num_classes=self.num_classes).float()
            logits = logits + self.persistence_alpha * prior
        return logits


class STFLATMLP(nn.Module):
    """Ablation A3: flatten spatial tensor (no CNN) + per-step MLP, then LSTM."""

    def __init__(
        self,
        in_channels: int,
        grid_h: int,
        grid_w: int,
        hidden_size: int = 64,
        num_classes: int = 4,
        dropout: float = 0.2,
        scalar_dim: int = 0,
    ):
        super().__init__()
        flat_dim = in_channels * grid_h * grid_w
        self.frame_mlp = nn.Sequential(
            nn.Linear(flat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
        )
        self.scalar_dim = int(scalar_dim)
        self.embed_dim = 32 + self.scalar_dim
        self.lstm = nn.LSTM(self.embed_dim, hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor, x_scalar: torch.Tensor | None = None) -> torch.Tensor:
        b, t, c, h, w = x.shape
        z = self.frame_mlp(x.reshape(b * t, c * h * w)).reshape(b, t, 32)
        if self.scalar_dim > 0:
            if x_scalar is None:
                raise ValueError("scalar_dim>0 but x_scalar is None")
            z = torch.cat([z, x_scalar], dim=-1)
        out, _ = self.lstm(z)
        return self.head(out[:, -1, :])


class STCNNOnly(nn.Module):
    """Ablation A4: CNN encoder + temporal mean pooling (no LSTM)."""

    def __init__(
        self,
        in_channels: int = 4,
        conv_channels: tuple[int, int] = (16, 32),
        num_classes: int = 4,
        dropout: float = 0.2,
        scalar_dim: int = 0,
    ):
        super().__init__()
        c1, c2 = conv_channels
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.scalar_dim = int(scalar_dim)
        self.embed_dim = c2 + self.scalar_dim
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.embed_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, x_scalar: torch.Tensor | None = None) -> torch.Tensor:
        b, t, c, h, w = x.shape
        flat = x.reshape(b * t, c, h, w)
        z_spatial = self.spatial_encoder(flat).reshape(b, t, -1)
        if self.scalar_dim > 0:
            if x_scalar is None:
                raise ValueError("scalar_dim>0 but x_scalar is None")
            z = torch.cat([z_spatial, x_scalar], dim=-1)
        else:
            z = z_spatial
        pooled = z.mean(dim=1)
        return self.head(pooled)


def fit_gtsep_dl(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    test_x: torch.Tensor,
    seed: int,
    n_states: int,
    train_cfg: dict,
    train_scalar: torch.Tensor | None = None,
    test_scalar: torch.Tensor | None = None,
    train_disturbance: torch.Tensor | None = None,
    test_disturbance: torch.Tensor | None = None,
    train_current_state: torch.Tensor | None = None,
    test_current_state: torch.Tensor | None = None,
) -> np.ndarray:
    """Train a spatio-temporal classifier and return softmax probabilities on test_x.

    train_x / test_x must already be standardized per-channel.
    train_scalar / test_scalar (optional): standardized scalar feature streams,
        shape (N, T, F), used only when the model has scalar_dim > 0.
    train_current_state / test_current_state (optional): per-sample current
        state class index, used only when the model has use_persistence_prior=True.
    train_cfg keys: learning_rate, batch_size, epochs, weight_decay (optional).
    """
    torch.manual_seed(seed)
    lr = float(train_cfg.get("learning_rate", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    epochs = int(train_cfg.get("epochs", 45))
    use_cosine = bool(train_cfg.get("cosine_schedule", True))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
        if use_cosine
        else None
    )
    train_counts = np.bincount(train_y.cpu().numpy(), minlength=n_states).astype(np.float32)
    class_weights = train_counts.sum() / np.maximum(train_counts, 1.0)
    class_weights = class_weights / max(1e-6, float(class_weights.mean()))
    loss_fn = FocalLoss(gamma=2.0, weight=torch.tensor(class_weights, dtype=torch.float32))
    batch_size = int(train_cfg.get("batch_size", 32))
    use_scalar = train_scalar is not None
    use_disturbance = train_disturbance is not None
    use_state = train_current_state is not None
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(train_x.shape[0])
        for start in range(0, train_x.shape[0], batch_size):
            idx = perm[start : start + batch_size]
            kwargs = {}
            if use_scalar:
                kwargs["x_scalar"] = train_scalar[idx]
            if use_disturbance:
                kwargs["disturbance"] = train_disturbance[idx]
            if use_state:
                kwargs["current_state"] = train_current_state[idx]
            logits = model(train_x[idx], **kwargs)
            loss = loss_fn(logits, train_y[idx])
            opt.zero_grad()
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            opt.step()
        if scheduler is not None:
            scheduler.step()
    model.eval()
    with torch.no_grad():
        kwargs = {}
        if use_scalar:
            kwargs["x_scalar"] = test_scalar
        if use_disturbance:
            kwargs["disturbance"] = test_disturbance
        if use_state:
            kwargs["current_state"] = test_current_state
        prob = torch.softmax(model(test_x, **kwargs), dim=1).cpu().numpy()
    return prob


def channel_standardize(
    train_tensors: np.ndarray,
    test_tensors: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Standardize each channel using stats from the train tensors only.

    train_tensors shape: (N, T, C, H, W) or (N, C, H, W).
    Returns (train_scaled, test_scaled_or_None, mean, std), both with per-channel stats.
    """
    if train_tensors.ndim == 5:
        axes = (0, 1, 3, 4)
    elif train_tensors.ndim == 4:
        axes = (0, 2, 3)
    else:
        raise ValueError(f"Unsupported tensor shape {train_tensors.shape}")
    mean = train_tensors.mean(axis=axes, keepdims=True).astype(np.float32)
    std = train_tensors.std(axis=axes, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    train_scaled = (train_tensors - mean) / std
    test_scaled = (test_tensors - mean) / std if test_tensors is not None else None
    return train_scaled.astype(np.float32), test_scaled, mean, std


def build_tensor_sequences(
    grid_tensors: np.ndarray,
    y: np.ndarray,
    seq_len: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slide a length-seq_len window over (N, C, H, W) tensors, label is y[end+horizon].

    Returns (seq_x, seq_y, end_positions), where:
        seq_x shape = (M, seq_len, C, H, W)
        seq_y shape = (M,)
        end_positions[i] = last window index in the i-th sequence
    """
    n = grid_tensors.shape[0]
    xs, ys, ends = [], [], []
    for end in range(seq_len - 1, n - horizon):
        xs.append(grid_tensors[end - seq_len + 1 : end + 1])
        ys.append(int(y[end + horizon]))
        ends.append(end)
    if not xs:
        c, h, w = grid_tensors.shape[1:]
        return (
            np.zeros((0, seq_len, c, h, w), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.stack(xs, axis=0).astype(np.float32),
        np.asarray(ys, dtype=np.int64),
        np.asarray(ends, dtype=np.int64),
    )
