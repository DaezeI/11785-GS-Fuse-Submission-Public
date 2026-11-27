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


class JointResidualHead(nn.Module):
    """Predict a correction from price context and encoded text jointly."""

    def __init__(self, price_dim, text_dim, output_dim, text_width=32,
                 hidden_width=32, hidden_layers=1):
        super().__init__()
        if hidden_layers not in (1, 2):
            raise ValueError("The joint decoder supports one or two hidden layers")
        self.text_projector = nn.Sequential(nn.Linear(text_dim, text_width), nn.GELU())
        layers = [nn.Linear(price_dim + text_width, hidden_width), nn.GELU()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_width, hidden_width), nn.GELU()])
        layers.append(nn.Linear(hidden_width, output_dim))
        self.joint_mlp = nn.Sequential(*layers)
        nn.init.zeros_(self.joint_mlp[-1].weight)
        nn.init.zeros_(self.joint_mlp[-1].bias)

    def forward(self, price_context, text):
        combined = torch.cat((price_context, self.text_projector(text)), dim=-1)
        return self.joint_mlp(combined)


class CompactReturn(nn.Module):
    def __init__(self, width=32, text_dim=0, output_dim=3, fusion="add",
                 joint_hidden_layers=1):
        super().__init__()
        if fusion not in ("add", "joint_mlp"):
            raise ValueError("Unknown fusion mode")
        if fusion == "joint_mlp" and not text_dim:
            raise ValueError("Joint decoding requires encoded text")
        self.fusion = fusion
        self.price_encoder = nn.Sequential(nn.Conv1d(4, width, 1), nn.GELU(),
            *[CausalBlock(width, dilation) for dilation in (1, 2, 4, 8)])
        self.price_head = nn.Sequential(nn.Linear(2 * width, 64), nn.GELU(), nn.Linear(64, output_dim))
        self.text_dim = text_dim
        if fusion == "joint_mlp":
            self.text_head = JointResidualHead(2 * width, text_dim, output_dim,
                                              hidden_layers=joint_hidden_layers)
        else:
            self.text_head = (nn.Sequential(nn.Linear(text_dim, 32), nn.GELU(), nn.Linear(32, output_dim))
                              if text_dim else None)
        # Start from zero-return and zero text correction; both are useful controls.
        nn.init.zeros_(self.price_head[-1].weight)
        nn.init.zeros_(self.price_head[-1].bias)
        if self.text_head is not None and fusion == "add":
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
            correction = (self.text_head(context, text) if self.fusion == "joint_mlp"
                          else self.text_head(text))
            result = result + correction
        return result
