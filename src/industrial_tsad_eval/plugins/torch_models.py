"""Torch model factories used by optional detector plugins."""

from __future__ import annotations

from typing import Any


def build_lstm_forecaster(
    torch: Any,
    *,
    feature_count: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
) -> Any:
    """Build a many-to-one LSTM next-step forecaster."""
    nn = torch.nn

    class LSTMForecaster(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=feature_count,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden_size, feature_count)

        def forward(self, x: Any) -> Any:
            output, _hidden = self.lstm(x)
            return self.head(output[:, -1, :])

    return LSTMForecaster()


def build_tcn_forecaster(
    torch: Any,
    *,
    feature_count: int,
    d: int,
) -> Any:
    """Build a compact TCN next-step forecaster for DRA detection."""
    nn = torch.nn

    class TCNForecaster(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(feature_count, d, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(d, d, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Linear(d, feature_count)

        def forward(self, x: Any) -> Any:
            encoded = self.net(x).squeeze(-1)
            return self.head(encoded)

    return TCNForecaster()


def build_hvae(
    torch: Any,
    *,
    feature_count: int,
    window: int,
    latent_dim: int,
) -> Any:
    """Build a compact HVAE-style reconstruction model for InterFusion."""
    nn = torch.nn
    functional = torch.nn.functional
    input_dim = window * feature_count
    hidden_dim = max(16, latent_dim * 4)

    class HVAE(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.mu = nn.Linear(hidden_dim, latent_dim)
            self.logvar = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x: Any) -> tuple[Any, Any, Any]:
            flat = x.reshape(x.shape[0], input_dim)
            hidden = self.encoder(flat)
            mu = self.mu(hidden)
            logvar = self.logvar(hidden).clamp(-8.0, 8.0)
            if self.training:
                std = torch.exp(0.5 * logvar)
                z = mu + torch.randn_like(std) * std
            else:
                z = mu
            reconstruction = self.decoder(z).reshape_as(x)
            return reconstruction, mu, logvar

        def negative_elbo(self, x: Any, kl_weight: float) -> Any:
            reconstruction, mu, logvar = self.forward(x)
            recon = functional.mse_loss(reconstruction, x)
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            return recon + kl_weight * kl

        def deterministic_score(self, x: Any) -> Any:
            reconstruction, _mu, _logvar = self.forward(x)
            return ((reconstruction - x) ** 2).mean(dim=(1, 2))

    return HVAE()


def build_drcad(
    torch: Any,
    *,
    feature_count: int,
    window: int,
    patch_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    mlp_dim: int,
    dropout: float,
) -> Any:
    """Build a dual-view contrastive detector approximating DRCAD blocks 1-2."""
    nn = torch.nn
    functional = torch.nn.functional
    patch_dim = patch_size * feature_count

    class DRCADNet(nn.Module):  # type: ignore[name-defined,misc]
        def __init__(self) -> None:
            super().__init__()
            self.n_patches = window // patch_size
            self.norm = nn.LayerNorm(feature_count)
            self.proj_in = nn.Linear(patch_dim, d_model)
            self.proj_pw = nn.Linear(patch_dim, d_model)
            self.in_layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=mlp_dim,
                        dropout=dropout,
                        batch_first=True,
                        activation="gelu",
                    )
                    for _ in range(n_layers)
                ]
            )
            self.pw_layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=n_heads,
                        dim_feedforward=mlp_dim,
                        dropout=dropout,
                        batch_first=True,
                        activation="gelu",
                    )
                    for _ in range(n_layers)
                ]
            )
            self.recon_head = nn.Sequential(
                nn.Linear(d_model * 2, mlp_dim),
                nn.GELU(),
                nn.Linear(mlp_dim, patch_dim),
            )

        def _patches(self, x: Any) -> Any:
            normalized = self.norm(x)
            return normalized.reshape(x.shape[0], self.n_patches, patch_dim)

        def forward(self, x: Any) -> tuple[Any, Any]:
            patches = self._patches(x)
            s_view = self.proj_in(patches)
            n_view = self.proj_pw(patches)
            for layer in self.in_layers:
                s_view = layer(s_view)
            for layer in self.pw_layers:
                n_view = layer(n_view)
            return functional.softmax(s_view, dim=-1), functional.softmax(n_view, dim=-1)

        def reconstruct(self, x: Any) -> Any:
            patches = self._patches(x)
            s_view = self.proj_in(patches)
            n_view = self.proj_pw(patches)
            for layer in self.in_layers:
                s_view = layer(s_view)
            for layer in self.pw_layers:
                n_view = layer(n_view)
            decoded = self.recon_head(torch.cat([s_view, n_view], dim=-1))
            return decoded.reshape_as(x)

        def contrastive_loss(self, x: Any) -> Any:
            s_prob, n_prob = self.forward(x)
            s_log = torch.log(s_prob + 1e-10)
            n_log = torch.log(n_prob + 1e-10)
            loss_s = functional.kl_div(s_log, n_prob.detach(), reduction="batchmean")
            loss_n = functional.kl_div(n_log, s_prob.detach(), reduction="batchmean")
            reconstruction = self.reconstruct(x)
            recon = functional.mse_loss(reconstruction, x)
            return loss_s + loss_n + recon

        def deterministic_score(self, x: Any) -> Any:
            s_prob, n_prob = self.forward(x)
            s_log = torch.log(s_prob + 1e-10)
            n_log = torch.log(n_prob + 1e-10)
            kl_sn = functional.kl_div(s_log, n_prob, reduction="none").sum(dim=-1).mean(dim=-1)
            kl_ns = functional.kl_div(n_log, s_prob, reduction="none").sum(dim=-1).mean(dim=-1)
            return kl_sn + kl_ns

    return DRCADNet()
