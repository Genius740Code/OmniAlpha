"""The Meridian value/policy CNN in torch, plus the exact export/load path for
the Rust `weights.rs` (26 tensors, 101,985 f32s, BatchNorm folded).

The training model uses BatchNorm and bias-free convs where BN follows; the
exported (folded) model is conv+bias only, identical to `nn.rs`'s forward:

    stem  conv5x5 + ReLU
    4 x   [conv3x3 + ReLU, conv3x3, add, ReLU]
    value conv1x1 + ReLU, global avg pool, linear, tanh
    policy conv3x3 + ReLU, conv1x1 -> flatten (channel-major == move id)
"""

import numpy as np
import torch
import torch.nn as nn

# Architecture, mirroring nn.rs (per-game adapter constants live there).
IN_PLANES, BOARD, POINTS = 14, 19, 361
CH, BLOCKS, PCH, VCH = 32, 4, 48, 16
NUM_MOVES = 48 * POINTS  # 17,328

# The 26 exported tensors, in the exact order nn::Net::decode reads them.
LAYOUT = ([CH * IN_PLANES * 25, CH]
          + [CH * CH * 9, CH, CH * CH * 9, CH] * BLOCKS
          + [VCH * CH, VCH, VCH, 1]
          + [PCH * CH * 9, PCH, PCH * PCH, PCH])
LAYOUT_NAMES = (["stem_w", "stem_b"]
               + [f"b{i}_{p}" for i in range(BLOCKS) for p in ("c1_w", "c1_b", "c2_w", "c2_b")]
               + ["v_w", "v_b", "v_fc_w", "v_fc_b", "p1_w", "p1_b", "p2_w", "p2_b"])
assert sum(LAYOUT) == 101_985 and len(LAYOUT) == 26

EPS = 1e-5  # torch BatchNorm2d default


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(CH, CH, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(CH)
        self.c2 = nn.Conv2d(CH, CH, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(CH)

    def forward(self, x):
        return torch.relu(x + self.bn2(self.c2(torch.relu(self.bn1(self.c1(x))))))


class TrainNet(nn.Module):
    """The model as trained: BatchNorm, bias-free convs where BN follows."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(IN_PLANES, CH, 5, padding=2, bias=False)
        self.stem_bn = nn.BatchNorm2d(CH)
        self.blocks = nn.ModuleList(Residual() for _ in range(BLOCKS))
        self.v = nn.Conv2d(CH, VCH, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(VCH)
        self.v_fc = nn.Linear(VCH, 1)
        self.p1 = nn.Conv2d(CH, PCH, 3, padding=1, bias=False)
        self.p1_bn = nn.BatchNorm2d(PCH)
        self.p2 = nn.Conv2d(PCH, PCH, 1, bias=True)  # logits head: no BN after

    def forward(self, x, policy=True):
        h = torch.relu(self.stem_bn(self.stem(x)))
        for block in self.blocks:
            h = block(h)
        v = torch.tanh(self.v_fc(
            torch.relu(self.v_bn(self.v(h))).flatten(2).mean(2))).squeeze(1)
        if not policy:
            return v, None
        p = self.p2(torch.relu(self.p1_bn(self.p1(h)))).flatten(1)
        return v, p


class FoldedNet(nn.Module):
    """The exported model: conv+bias only, byte-compatible with nn.rs."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(IN_PLANES, CH, 5, padding=2, bias=True)
        self.blocks = nn.ModuleList(nn.ModuleDict({
            "c1": nn.Conv2d(CH, CH, 3, padding=1, bias=True),
            "c2": nn.Conv2d(CH, CH, 3, padding=1, bias=True),
        }) for _ in range(BLOCKS))
        self.v = nn.Conv2d(CH, VCH, 1, bias=True)
        self.v_fc = nn.Linear(VCH, 1)
        self.p1 = nn.Conv2d(CH, PCH, 3, padding=1, bias=True)
        self.p2 = nn.Conv2d(PCH, PCH, 1, bias=True)

    def forward(self, x, policy=True):
        h = torch.relu(self.stem(x))
        for block in self.blocks:
            h = torch.relu(h + block["c2"](torch.relu(block["c1"](h))))
        v = torch.tanh(self.v_fc(
            torch.relu(self.v(h)).flatten(2).mean(2))).squeeze(1)
        if not policy:
            return v, None
        return v, self.p2(torch.relu(self.p1(h))).flatten(1)


@torch.no_grad()
def fold(model: TrainNet) -> dict:
    """Folds running BatchNorm statistics into the conv weights/biases."""
    model.eval()
    def fold_conv(conv, bn):
        scale = bn.weight / torch.sqrt(bn.running_var + EPS)
        w = conv.weight * scale.view(-1, 1, 1, 1)
        b = bn.running_mean * -scale + bn.bias
        return w, b

    out = {}
    w, out["stem_b"] = fold_conv(model.stem, model.stem_bn)
    out["stem_w"] = w
    for i, block in enumerate(model.blocks):
        w, out[f"b{i}_c1_b"] = fold_conv(block.c1, block.bn1)
        out[f"b{i}_c1_w"] = w
        w, out[f"b{i}_c2_b"] = fold_conv(block.c2, block.bn2)
        out[f"b{i}_c2_w"] = w
    w, out["v_b"] = fold_conv(model.v, model.v_bn)
    out["v_w"] = w
    out["v_fc_w"] = model.v_fc.weight.detach().reshape(-1)
    out["v_fc_b"] = model.v_fc.bias.detach().reshape(-1)
    w, out["p1_b"] = fold_conv(model.p1, model.p1_bn)
    out["p1_w"] = w
    out["p2_w"], out["p2_b"] = model.p2.weight.detach(), model.p2.bias.detach()
    return out


def folded_state_dict_from_bytes(tensors: list) -> dict:
    """26 f32 tensors (decode order) -> a FoldedNet state_dict."""
    assert len(tensors) == 26
    conv = lambda t, shape: torch.from_numpy(np.ascontiguousarray(t.reshape(shape)))
    sd = {"v_fc.weight": conv(tensors[20], (1, VCH)), "v_fc.bias": conv(tensors[21], (1,))}
    sd["stem.weight"], sd["stem.bias"] = conv(tensors[0], (CH, IN_PLANES, 5, 5)), conv(tensors[1], (CH,))
    for i in range(BLOCKS):
        base = 2 + 4 * i
        sd[f"blocks.{i}.c1.weight"] = conv(tensors[base], (CH, CH, 3, 3))
        sd[f"blocks.{i}.c1.bias"] = conv(tensors[base + 1], (CH,))
        sd[f"blocks.{i}.c2.weight"] = conv(tensors[base + 2], (CH, CH, 3, 3))
        sd[f"blocks.{i}.c2.bias"] = conv(tensors[base + 3], (CH,))
    sd["v.weight"], sd["v.bias"] = conv(tensors[18], (VCH, CH, 1, 1)), conv(tensors[19], (VCH,))
    sd["p1.weight"], sd["p1.bias"] = conv(tensors[22], (PCH, CH, 3, 3)), conv(tensors[23], (PCH,))
    sd["p2.weight"], sd["p2.bias"] = conv(tensors[24], (PCH, PCH, 1, 1)), conv(tensors[25], (PCH,))
    return sd


def export_f32(folded: dict) -> np.ndarray:
    """The folded tensors, concatenated little-endian f32 in LAYOUT order."""
    parts = [None] * 26
    parts[0], parts[1] = folded["stem_w"], folded["stem_b"]
    for i in range(BLOCKS):
        parts[2 + 4 * i:6 + 4 * i] = (folded[f"b{i}_c1_w"], folded[f"b{i}_c1_b"],
                                     folded[f"b{i}_c2_w"], folded[f"b{i}_c2_b"])
    parts[18], parts[19] = folded["v_w"], folded["v_b"]
    parts[20], parts[21] = folded["v_fc_w"], folded["v_fc_b"]
    parts[22], parts[23], parts[24], parts[25] = folded["p1_w"], folded["p1_b"], folded["p2_w"], folded["p2_b"]
    flat = [t.detach().cpu().numpy().astype("<f4").ravel() if torch.is_tensor(t)
            else np.asarray(t, "<f4").ravel() for t in parts]
    for name, t, want in zip(LAYOUT_NAMES, flat, LAYOUT):
        assert t.size == want, f"{name}: {t.size} != {want}"
    return np.concatenate(flat)


def parse_weights_rs(path: str) -> list:
    """Reads the f32 BYTES array out of a weights.rs file."""
    import re
    text = open(path).read()
    body = text[text.index("pub const BYTES"): text.index("];", text.index("pub const BYTES"))]
    hexes = re.findall(r"0x([0-9a-f]{2})", body)
    arr = np.frombuffer(bytes(int(h, 16) for h in hexes), dtype="u1")
    assert arr.size == sum(LAYOUT) * 4, f"{path}: {arr.size} bytes"
    return [t.copy() for t in np.split(arr.view("<f4"), np.cumsum(LAYOUT)[:-1])]


def load_folded_from_rs(path: str) -> FoldedNet:
    net = FoldedNet()
    net.load_state_dict(folded_state_dict_from_bytes(parse_weights_rs(path)))
    net.eval()
    return net


def folded_model_from_flat(flat: dict) -> FoldedNet:
    """A `fold()` dict -> a live FoldedNet, via the canonical export bytes."""
    arr = export_f32(flat)
    tensors = [t.copy() for t in np.split(arr, np.cumsum(LAYOUT)[:-1])]
    net = FoldedNet()
    net.load_state_dict(folded_state_dict_from_bytes(tensors))
    net.eval()
    return net


def write_weights_rs(path: str, flat: np.ndarray, header_lines: list) -> None:
    """Regenerates a weights.rs with the same LAYOUT and format as the original."""
    b = flat.astype("<f4").tobytes()
    assert len(b) == sum(LAYOUT) * 4
    with open(path, "w") as fh:
        fh.write(
            "//! Trained parameters of the Meridian value/policy CNN, as raw little-endian\n"
            "//! f32 bytes. GENERATED by bots/meridian/nn/train/export_weights.py | "
            + " | ".join(header_lines) + "\n"
            "//! do not edit by hand.\n"
            "//!\n"
            "//! `LAYOUT` lists the tensor lengths in f32 units, in the exact order\n"
            "//! `nn::Net::decode` reads them: stem, 4 residual blocks (c1_w, c1_b,\n"
            "//! c2_w, c2_b each), value head, policy head. BatchNorm is folded into\n"
            "//! the convolutions, so inference needs no statistics.\n\n")
    with open(path, "a") as fh:
        fh.write("/// Tensor lengths in f32 units, in decode order. Sums to 101,985 floats.\n")
        rows = [", ".join(f"{n:,}".replace(",", "_") for n in LAYOUT[i:i + 8]) for i in range(0, 26, 8)]
        fh.write("pub const LAYOUT: &[usize] = &[\n" + "".join(f"    {r},\n" for r in rows) + "];\n\n")
        fh.write("/// All tensors concatenated, little-endian f32. 101,985 * 4 = 407,940 bytes.\n")
        fh.write("pub const BYTES: &[u8] = &[\n")
        for i in range(0, len(b), 100):
            fh.write("    " + ", ".join(f"0x{x:02x}" for x in b[i:i + 100]) + ",\n")
        fh.write("];\n")
