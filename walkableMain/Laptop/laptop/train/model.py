"""Stage 6: 1D CNN that maps IMU samples to per-sample heel-strike probability.

Fully convolutional: input (batch, 6, T) -> logits (batch, 2, T), one
channel per leg, same length T. Dilated residual blocks give a receptive
field of 127 samples (1.27 s at 100 Hz) centred on each sample; live mode
(Stage 7) calls a step live.lookahead_s (0.3 s) after it happens.
"""

import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, ch, dilation, k=5):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.net = nn.Sequential(
            nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation), nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, k, padding=pad, dilation=dilation), nn.BatchNorm1d(ch),
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class StepCNN(nn.Module):
    def __init__(self, in_ch=6, ch=32, out_ch=2, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(in_ch, ch, 7, padding=3), nn.BatchNorm1d(ch), nn.ReLU())
        self.blocks = nn.Sequential(*[ResBlock(ch, d) for d in dilations])
        self.head = nn.Conv1d(ch, out_ch, 1)
        # stem adds 6; each block has two k=5 convs adding 4*d each
        self.receptive_field = 7 + sum(2 * 4 * d for d in dilations)  # samples

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


def predict(model, x_norm, device="cpu"):
    """x_norm (T, 6) normalized -> probabilities (T, 2)."""
    model.eval()
    with torch.no_grad():
        xt = torch.as_tensor(x_norm.T[None], dtype=torch.float32, device=device)
        return torch.sigmoid(model(xt))[0].T.cpu().numpy()


def save(path, model, norm, cfg_train, extra=None):
    torch.save({"state_dict": model.state_dict(), "norm": norm.to_dict(),
                "channels": cfg_train["channels"], "train_cfg": cfg_train, **(extra or {})}, path)


def load(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = StepCNN(ch=ck["channels"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck
