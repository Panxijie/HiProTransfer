from pathlib import Path

import numpy as np
import torch
from torch import nn, optim
from tqdm import tqdm

from .data import make_loader


def _group_count(num_channels):
    for groups in (8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


class ResidualTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        groups = _group_count(out_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.act = nn.LeakyReLU(0.2)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.residual is None else self.residual(x)
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class MaskAwareTemporalEncoder(nn.Module):
    def __init__(self, feature_dim, latent_dim, window_size, hidden_channels=32, kernel_size=5):
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.feature_dim = int(feature_dim)
        self.window_size = int(window_size)
        self.input_projection = nn.Conv1d(self.feature_dim * 2, hidden_channels, 1)
        self.temporal = nn.ModuleList(
            [ResidualTemporalBlock(hidden_channels, hidden_channels, kernel_size, dilation=1)]
        )
        self.feature_hidden_dim = 16
        self.feature_projector = nn.Linear(hidden_channels, self.feature_dim * self.feature_hidden_dim)
        self.norm = nn.LayerNorm(self.feature_hidden_dim)
        self.bottleneck = nn.Linear(self.feature_hidden_dim, latent_dim)

    def _prepare_mask(self, x, mask):
        if mask is None:
            return torch.ones(x.size(0), self.feature_dim, device=x.device, dtype=x.dtype)
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand(x.size(0), -1)
        return mask.to(device=x.device, dtype=x.dtype)

    def forward(self, x, mask=None):
        mask = self._prepare_mask(x, mask)
        x_masked = x * mask.unsqueeze(1)
        value_channels = x_masked.transpose(1, 2)
        mask_channels = mask.unsqueeze(-1).expand(-1, self.feature_dim, self.window_size)
        h = torch.cat([value_channels, mask_channels], dim=1)
        h = self.input_projection(h)
        for block in self.temporal:
            h = block(h)
        h = h.mean(dim=-1)
        tokens = self.feature_projector(h).view(-1, self.feature_dim, self.feature_hidden_dim)
        tokens = tokens * mask.unsqueeze(-1)
        pooled = tokens.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        return self.bottleneck(self.norm(pooled))


class TemporalDecoder(nn.Module):
    def __init__(self, latent_dim, feature_dim, window_size):
        super().__init__()
        output_dim = int(feature_dim) * int(window_size)
        hidden1 = max(output_dim // 4, latent_dim)
        hidden2 = max(output_dim // 2, latent_dim)
        self.feature_dim = int(feature_dim)
        self.window_size = int(window_size)
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden1),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden1, hidden2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden2, output_dim),
        )

    def forward(self, z):
        return self.net(z).view(-1, self.window_size, self.feature_dim)


class MATA(object):
    """Mask-aware Adversarial Temporal Autoencoder."""

    def __init__(
        self,
        feature_dim,
        window_size,
        latent_dim=30,
        batch_size=128,
        hidden_channels=32,
        kernel_size=5,
        lr=1e-3,
        device=None,
    ):
        self.feature_dim = int(feature_dim)
        self.window_size = int(window_size)
        self.latent_dim = int(latent_dim)
        self.batch_size = int(batch_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.encoder = MaskAwareTemporalEncoder(
            self.feature_dim,
            self.latent_dim,
            self.window_size,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
        ).to(self.device)
        self.decoder_primary = TemporalDecoder(self.latent_dim, self.feature_dim, self.window_size).to(self.device)
        self.decoder_refiner = TemporalDecoder(self.latent_dim, self.feature_dim, self.window_size).to(self.device)
        self.loss_func = nn.MSELoss(reduction="none")
        self.optimizer_primary = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder_primary.parameters()),
            lr=lr,
        )
        self.optimizer_refiner = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder_refiner.parameters()),
            lr=lr,
        )

    def _prepare_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            x, mask = batch
        else:
            x, mask = batch, None
        x = x.to(self.device).float()
        if mask is None:
            mask = torch.ones(x.size(0), self.feature_dim, device=self.device)
        else:
            mask = mask.to(self.device).float()
        return x, mask

    def _masked_mean(self, error, mask):
        error = error * mask.unsqueeze(1)
        denom = (mask.sum(dim=1) * self.window_size).clamp(min=1.0)
        return error.sum(dim=(1, 2)) / denom

    def _losses(self, x, mask, epoch, prototype=None, proto_reg_strength=0.0):
        z = self.encoder(x, mask)
        rec_primary = self.decoder_primary(z)
        rec_refiner = self.decoder_refiner(z)
        refined_primary = self.decoder_refiner(self.encoder(rec_primary, mask))

        err_primary = self.loss_func(rec_primary, x)
        err_refined = self.loss_func(refined_primary, x)
        err_refiner = self.loss_func(rec_refiner, x)
        loss_primary = (1.0 / epoch) * self._masked_mean(err_primary, mask).mean()
        loss_primary = loss_primary + (1.0 - 1.0 / epoch) * self._masked_mean(err_refined, mask).mean()
        loss_refiner = (1.0 / epoch) * self._masked_mean(err_refiner, mask).mean()
        loss_refiner = loss_refiner - (1.0 - 1.0 / epoch) * self._masked_mean(err_refined, mask).mean()

        if prototype is not None and proto_reg_strength > 0:
            proto = torch.as_tensor(prototype, device=self.device, dtype=z.dtype)
            if proto.dim() == 1:
                proto = proto.unsqueeze(0).expand(z.size(0), -1)
            reg = ((z - proto) ** 2).mean()
            loss_primary = loss_primary + proto_reg_strength * reg
            loss_refiner = loss_refiner + proto_reg_strength * reg
        return loss_primary, loss_refiner

    def fit(
        self,
        values,
        epochs,
        metric_masks=None,
        valid_values=None,
        valid_metric_masks=None,
        prototype=None,
        proto_reg_strength=0.0,
        progress_desc="MATA training",
    ):
        train_loader = make_loader(
            values,
            window_size=self.window_size,
            batch_size=self.batch_size,
            masks=metric_masks,
            shuffle=True,
            drop_last=False,
        )
        self.encoder.train()
        self.decoder_primary.train()
        self.decoder_refiner.train()
        for epoch in tqdm(range(1, int(epochs) + 1), desc=progress_desc, unit="epoch"):
            for batch in train_loader:
                x, mask = self._prepare_batch(batch)
                self.optimizer_primary.zero_grad()
                self.optimizer_refiner.zero_grad()
                loss_primary, loss_refiner = self._losses(
                    x,
                    mask,
                    epoch,
                    prototype=prototype,
                    proto_reg_strength=proto_reg_strength,
                )
                loss_primary.backward(retain_graph=True)
                loss_refiner.backward()
                self.optimizer_primary.step()
                self.optimizer_refiner.step()

    def score(self, values, metric_mask=None):
        loader = make_loader(
            values,
            window_size=self.window_size,
            batch_size=self.batch_size,
            masks=metric_mask,
            shuffle=False,
            drop_last=False,
        )
        scores = []
        self.encoder.eval()
        self.decoder_primary.eval()
        self.decoder_refiner.eval()
        with torch.no_grad():
            for batch in loader:
                x, mask = self._prepare_batch(batch)
                z = self.encoder(x, mask)
                rec_primary = self.decoder_primary(z)
                refined_primary = self.decoder_refiner(self.encoder(rec_primary, mask))
                err_primary = self._masked_mean((rec_primary - x) ** 2, mask)
                err_refined = self._masked_mean((refined_primary - x) ** 2, mask)
                scores.append((0.5 * err_primary + 0.5 * err_refined).cpu().numpy())
        return np.concatenate(scores, axis=0).astype(np.float32)

    def latent(self, values, metric_mask=None):
        loader = make_loader(values, self.window_size, self.batch_size, masks=metric_mask)
        chunks = []
        self.encoder.eval()
        with torch.no_grad():
            for batch in loader:
                x, mask = self._prepare_batch(batch)
                chunks.append(self.encoder(x, mask).cpu().numpy())
        return np.concatenate(chunks, axis=0).astype(np.float32)

    def save(self, model_dir):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.encoder.state_dict(), model_dir / "mata_encoder.pt")
        torch.save(self.decoder_primary.state_dict(), model_dir / "mata_decoder_primary.pt")
        torch.save(self.decoder_refiner.state_dict(), model_dir / "mata_decoder_refiner.pt")

    def restore(self, model_dir):
        model_dir = Path(model_dir)
        self.encoder.load_state_dict(torch.load(model_dir / "mata_encoder.pt", map_location=self.device))
        self.decoder_primary.load_state_dict(torch.load(model_dir / "mata_decoder_primary.pt", map_location=self.device))
        self.decoder_refiner.load_state_dict(torch.load(model_dir / "mata_decoder_refiner.pt", map_location=self.device))
