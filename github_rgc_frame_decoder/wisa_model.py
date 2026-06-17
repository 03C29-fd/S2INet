import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, n_cells, latent_dim=512, channels=128, dropout=0.15):
        super().__init__()
        self.n_cells = n_cells
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(n_cells, channels, kernel_size=1, padding=0),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv1d(n_cells, channels, kernel_size=3, padding=1, dilation=1),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv1d(n_cells, channels, kernel_size=3, padding=2, dilation=2),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv1d(n_cells, channels, kernel_size=5, padding=4, dilation=2),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                ),
            ]
        )
        self.fuse = nn.Sequential(
            nn.Linear(channels * len(self.branches), latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Sequential(nn.Linear(latent_dim, latent_dim), nn.Sigmoid())

    def forward(self, x):
        # x: [B, history, cells]
        x = x.transpose(1, 2)
        pooled = []
        for branch in self.branches:
            out = branch(x)
            pooled.append(out.mean(dim=-1))
        latent = self.fuse(torch.cat(pooled, dim=1))
        return latent * self.gate(latent)


class AttentionTemporalEncoder(nn.Module):
    def __init__(
        self,
        n_cells,
        latent_dim=512,
        channels=128,
        dropout=0.15,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.input_proj = nn.Linear(n_cells, channels)
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=num_heads,
            dim_feedforward=channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.multiscale = MultiScaleTemporalEncoder(n_cells, latent_dim, channels, dropout)
        self.attn_pool = nn.Sequential(nn.Linear(channels, 1), nn.Softmax(dim=1))
        self.fuse = nn.Sequential(
            nn.Linear(latent_dim + channels, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: [B, history, cells]
        tokens = self.input_proj(x)
        tokens = self.transformer(tokens)
        weights = self.attn_pool(tokens)
        temporal_context = (tokens * weights).sum(dim=1)
        multiscale_context = self.multiscale(x)
        return self.fuse(torch.cat([multiscale_context, temporal_context], dim=1))


class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)
        channel_weight = torch.sigmoid(self.channel_mlp(avg) + self.channel_mlp(mx)).view(x.shape[0], x.shape[1], 1, 1)
        x = x * channel_weight
        spatial_input = torch.cat([x.mean(dim=1, keepdim=True), x.max(dim=1, keepdim=True).values], dim=1)
        spatial_weight = torch.sigmoid(self.spatial(spatial_input))
        return x * spatial_weight


class AttentionSpatialFrameDecoder(nn.Module):
    def __init__(self, latent_dim=512, image_size=(64, 64), base_channels=128, dropout=0.15):
        super().__init__()
        self.image_size = image_size
        self.seed_size = 8
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, base_channels * self.seed_size * self.seed_size),
            nn.GELU(),
        )
        self.block1 = self._block(base_channels, base_channels, dropout)
        self.block2 = self._block(base_channels, base_channels // 2, dropout)
        self.block3 = self._block(base_channels // 2, base_channels // 4, dropout)
        self.out = nn.Conv2d(base_channels // 4, 1, kernel_size=3, padding=1)

    @staticmethod
    def _block(in_channels, out_channels, dropout):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            ChannelSpatialAttention(out_channels),
            nn.Dropout2d(dropout * 0.5),
        )

    def forward(self, latent):
        out = self.fc(latent)
        out = out.reshape(latent.shape[0], -1, self.seed_size, self.seed_size)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.out(out)
        if out.shape[-2:] != self.image_size:
            out = F.interpolate(out, size=self.image_size, mode="bilinear", align_corners=False)
        return out


class SpatialFrameDecoder(nn.Module):
    def __init__(self, latent_dim=512, image_size=(64, 64), base_channels=128, dropout=0.15):
        super().__init__()
        self.image_size = image_size
        self.seed_size = 8
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, base_channels * self.seed_size * self.seed_size),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Dropout2d(dropout * 0.5),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.GELU(),
            nn.Dropout2d(dropout * 0.5),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels // 2, base_channels // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels // 4),
            nn.GELU(),
            nn.Conv2d(base_channels // 4, 1, kernel_size=3, padding=1),
        )

    def forward(self, latent):
        out = self.fc(latent)
        out = out.reshape(latent.shape[0], -1, self.seed_size, self.seed_size)
        out = self.decoder(out)
        if out.shape[-2:] != self.image_size:
            out = F.interpolate(out, size=self.image_size, mode="bilinear", align_corners=False)
        return out


class WISALiteDecoder(nn.Module):
    def __init__(
        self,
        n_cells,
        history_bins,
        image_size=(64, 64),
        latent_dim=512,
        temporal_channels=128,
        base_channels=128,
        dropout=0.15,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.history_bins = history_bins
        self.encoder = MultiScaleTemporalEncoder(n_cells, latent_dim, temporal_channels, dropout)
        self.decoder = SpatialFrameDecoder(latent_dim, image_size, base_channels, dropout)

    def forward(self, x, batchsize=None):
        if x.ndim == 2:
            x = x.reshape(x.shape[0], self.history_bins, self.n_cells)
        return self.decoder(self.encoder(x))


class WISAAttentionDecoder(nn.Module):
    def __init__(
        self,
        n_cells,
        history_bins,
        image_size=(64, 64),
        latent_dim=512,
        temporal_channels=128,
        base_channels=128,
        dropout=0.15,
        num_heads=4,
        num_layers=2,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.history_bins = history_bins
        self.encoder = AttentionTemporalEncoder(
            n_cells,
            latent_dim=latent_dim,
            channels=temporal_channels,
            dropout=dropout,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.decoder = AttentionSpatialFrameDecoder(latent_dim, image_size, base_channels, dropout)

    def forward(self, x, batchsize=None):
        if x.ndim == 2:
            x = x.reshape(x.shape[0], self.history_bins, self.n_cells)
        return self.decoder(self.encoder(x))
