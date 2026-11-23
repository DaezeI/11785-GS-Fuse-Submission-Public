"""Small endpoint-return forecaster with an independently frozen price baseline.

No pretrained backbone is loaded. Optional pooled text features are computed
outside the training loop; only their small residual head learns here.
"""
import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(width, width, 3, dilation=dilation)

    def forward(self, x):
        return x + F.gelu(self.conv(F.pad(x, (self.padding, 0))))


class CompactReturn(nn.Module):
    def __init__(self, width=32, text_dim=0, output_dim=3):
        super().__init__()
        self.price_encoder = nn.Sequential(nn.Conv1d(4, width, 1), nn.GELU(),
            *[CausalBlock(width, dilation) for dilation in (1, 2, 4, 8)])
        self.price_head = nn.Sequential(nn.Linear(2 * width, 64), nn.GELU(), nn.Linear(64, output_dim))
        self.text_dim = text_dim
        self.text_head = (nn.Sequential(nn.Linear(text_dim, 32), nn.GELU(), nn.Linear(32, output_dim))
                          if text_dim else None)
        # Start from zero-return and zero text correction; both are useful controls.
        nn.init.zeros_(self.price_head[-1].weight)
        nn.init.zeros_(self.price_head[-1].bias)
        if self.text_head is not None:
            nn.init.zeros_(self.text_head[-1].weight)
            nn.init.zeros_(self.text_head[-1].bias)

    def freeze_price(self):
        for module in (self.price_encoder, self.price_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, history, text=None):
        if history.ndim != 3 or history.shape[1] != 4:
            raise ValueError("Expected history [batch, 4 OHLC channels, observed bars]")
        tokens = self.price_encoder(history)
        context = torch.cat((tokens[:, :, -1], tokens.mean(-1)), dim=-1)
        result = self.price_head(context)
        if text is not None:
            if self.text_head is None or text.shape != (len(history), self.text_dim):
                raise ValueError("Text features do not match the configured residual head")
            result = result + self.text_head(text)
        return result
