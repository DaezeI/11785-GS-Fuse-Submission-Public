"""128d, four-layer price Transformer with one shared forecast-query parameter."""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class PredictionTokenTransformer(nn.Module):
    def __init__(self, d_model=128, depth=4, nhead=4, dim_feedforward=512,
                 dropout=.1, future_minutes=(5, 10, 15, 20, 25, 30),
                 activation_checkpointing=False, attention_batch_chunk=0):
        super().__init__()
        if d_model % 2 or d_model % nhead or depth < 1:
            raise ValueError("Hidden width must be even and divisible by attention heads")
        self.d_model = d_model
        self.activation_checkpointing = activation_checkpointing
        self.attention_batch_chunk = attention_batch_chunk
        self.input_embedding = nn.Linear(4, d_model)
        self.prediction_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("time_frequencies", torch.exp(
            -math.log(10000.) * torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("future_minutes", torch.tensor(future_minutes, dtype=torch.float32))
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout, activation="gelu",
                batch_first=True, norm_first=True) for _ in range(depth)])
        self.final_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, 1)

    def time_embedding(self, minutes):
        # Position units are five minutes; negative/overnight UTC offsets are valid.
        phase = (minutes.float().unsqueeze(-1) / 5) * self.time_frequencies
        return torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)

    @staticmethod
    def attention_mask(past_length, future_length, device):
        # PyTorch Transformer masks use True for a blocked query/key pair.
        mask = torch.zeros(past_length + future_length, past_length + future_length,
                           dtype=torch.bool, device=device)
        mask[:past_length, past_length:] = True
        return mask

    def encode(self, history, past_minutes, future_minutes=None):
        if history.ndim != 3 or history.shape[1] != 4:
            raise ValueError("Expected history [batch, 4, past_length]")
        batch, _, length = history.shape
        if past_minutes.shape != (batch, length):
            raise ValueError("Past timestamp offsets must match [batch, past_length]")
        offsets = self.future_minutes if future_minutes is None else future_minutes
        if offsets.ndim != 1 or len(offsets) == 0:
            raise ValueError("Expected a nonempty vector of future timestamp offsets")
        past = self.input_embedding(history.transpose(1, 2))
        past = past + self.time_embedding(past_minutes).to(past.dtype)
        future = (self.prediction_token.to(past.dtype).expand(batch, len(offsets), -1)
                  + self.time_embedding(offsets).to(past.dtype).unsqueeze(0))
        hidden = torch.cat((past, future), dim=1)
        mask = self.attention_mask(length, len(offsets), hidden.device)
        for layer in self.layers:
            # Batch-axis tiling is optional kernel/workspace control. A single
            # loss/backward/optimizer update still uses the entire logical batch.
            size = self.attention_batch_chunk or len(hidden)
            pieces = []
            for part in hidden.split(size):
                if self.activation_checkpointing and self.training and torch.is_grad_enabled():
                    pieces.append(checkpoint(layer, part, src_mask=mask, use_reentrant=False))
                else:
                    pieces.append(layer(part, src_mask=mask))
            hidden = torch.cat(pieces, dim=0) if len(pieces) > 1 else pieces[0]
        return self.final_norm(hidden)

    def forward(self, history, past_minutes, future_minutes=None):
        hidden = self.encode(history, past_minutes, future_minutes)
        return self.readout(hidden[:, history.shape[-1]:]).squeeze(-1)
