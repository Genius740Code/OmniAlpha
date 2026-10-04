"""Configurable policy/value networks. UNTESTED (no torch in the authoring sandbox) -- run `python -m gai.model` first."""
import torch, torch.nn as nn, torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, planes, h, w, actions, hidden=128, layers=2):
        super().__init__(); d = planes * h * w; mods = []
        for _ in range(layers): mods += [nn.Linear(d, hidden), nn.ReLU()]; d = hidden
        self.body = nn.Sequential(nn.Flatten(), *mods); self.pol = nn.Linear(d, actions); self.val = nn.Linear(d, 1)
    def forward(self, x):
        z = self.body(x); return self.pol(z), torch.tanh(self.val(z)).squeeze(-1)

class Block(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(c)
    def forward(self, x):
        y = F.relu(self.b1(self.c1(x))); y = self.b2(self.c2(y)); return F.relu(x + y)

class ResNet(nn.Module):
    def __init__(self, planes, h, w, actions, channels=64, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(planes, channels, 3, padding=1, bias=False), nn.BatchNorm2d(channels), nn.ReLU())
        self.blocks = nn.Sequential(*[Block(channels) for _ in range(blocks)])
        self.pconv = nn.Sequential(nn.Conv2d(channels, 2, 1), nn.BatchNorm2d(2), nn.ReLU()); self.pfc = nn.Linear(2 * h * w, actions)
        self.vconv = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.BatchNorm2d(1), nn.ReLU()); self.vfc1 = nn.Linear(h * w, 64); self.vfc2 = nn.Linear(64, 1)
    def forward(self, x):
        z = self.blocks(self.stem(x)); p = self.pfc(self.pconv(z).flatten(1))
        v = torch.tanh(self.vfc2(F.relu(self.vfc1(self.vconv(z).flatten(1))))).squeeze(-1); return p, v

def build(cfg, planes, h, w, actions):
    t = cfg.get("type", "mlp"); kw = {k: v for k, v in cfg.items() if k != "type"}
    return (MLP if t == "mlp" else ResNet)(planes, h, w, actions, **kw)

if __name__ == "__main__":  # smoke test
    for c in [{"type": "mlp"}, {"type": "resnet", "channels": 16, "blocks": 2}]:
        m = build(c, 2, 6, 7, 7); p, v = m(torch.zeros(3, 2, 6, 7)); assert p.shape == (3, 7) and v.shape == (3,); print(c["type"], "ok")
