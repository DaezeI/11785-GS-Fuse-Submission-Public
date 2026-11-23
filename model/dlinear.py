"""DLinear architecture shared by the continuous-window and event benchmarks."""

from torch import cat, nn


class SeriesDecomp(nn.Module):
    """Split a channel-first series into a moving-average trend and residual."""

    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x):
        front = x[:, :, :1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, self.kernel_size // 2)
        trend = self.avg(cat([front, x, end], dim=-1))
        return x - trend, trend


class DLinearBaseline(nn.Module):
    """Fit separate temporal projections for the trend and residual."""

    def __init__(self, seq_len, pred_len, d_feat=4, kernel_size=25):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.decomp = SeriesDecomp(kernel_size)
        self.linear_seasonal = nn.Linear(seq_len, pred_len)
        self.linear_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        seasonal, trend = self.decomp(x)
        return self.linear_seasonal(seasonal) + self.linear_trend(trend)
