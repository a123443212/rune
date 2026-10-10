import torch
import torch.nn as nn

from training.models.resnet_blocks import ConvBlock, PolicyHead, ValueHead

ARCH_ID = "RUNE-RESNET-01"
ARCH_VERSION = "0.1.0"


class ResnetTower(nn.Module):
    def __init__(self, board=19, channels=64, blocks=6, policy_size=None, value_h2=256, in_planes=1, feature_set="go_planes_v01"):
        super().__init__()
        self.board = board
        self.channels = channels
        self.blocks = blocks
        self.in_planes = in_planes
        self.feature_set = feature_set
        self.policy_size = policy_size if policy_size is not None else board * board + 1
        self.stem = nn.Conv2d(in_planes, channels, 3, padding=1, bias=True)
        self.tower = nn.ModuleList([ConvBlock(channels) for _ in range(blocks)])
        self.value = ValueHead(channels, value_h2)
        self.policy = PolicyHead(channels, board, self.policy_size)

    def forward(self, planes):
        x = torch.relu(self.stem(planes))
        for blk in self.tower:
            x = blk(x)
        pooled = x.mean(dim=(2, 3))
        flat = x.reshape(x.size(0), -1)
        value, wdl = self.value(pooled)
        logits, probs = self.policy(flat)
        return value, wdl, logits, probs

    def arch_tensors(self):
        d = {
            "stem_w": self.stem.weight.detach(),
            "stem_b": self.stem.bias.detach(),
        }
        for i, blk in enumerate(self.tower):
            d[f"b{i}_w1"] = blk.c1.weight.detach()
            d[f"b{i}_b1"] = blk.c1.bias.detach()
            d[f"b{i}_w2"] = blk.c2.weight.detach()
            d[f"b{i}_b2"] = blk.c2.bias.detach()
        d["vh1"] = self.value.fc1.weight.detach()
        d["bh1"] = self.value.fc1.bias.detach()
        d["wv"] = self.value.fcv.weight.detach().reshape(-1)
        d["bv"] = self.value.fcv.bias.detach()
        d["wwdl"] = self.value.fcwdl.weight.detach()
        d["bwdl"] = self.value.fcwdl.bias.detach()
        d["wpol"] = self.policy.fc.weight.detach()
        d["bpol"] = self.policy.fc.bias.detach()
        return d

    def export_order(self):
        order = ["stem_w", "stem_b"]
        for i in range(len(self.tower)):
            order += [f"b{i}_w1", f"b{i}_b1", f"b{i}_w2", f"b{i}_b2"]
        return order + ["vh1", "bh1", "wv", "bv", "wwdl", "bwdl", "wpol", "bpol"]

    def embedding_tensors(self):
        return {}

    def model_spec(self, game="go", quantization="fp32"):
        return {
            "game": game,
            "arch": ARCH_ID,
            "arch_version": ARCH_VERSION,
            "feature_set": self.feature_set,
            "tokens": self.board,
            "token_dim": self.channels,
            "board_size": self.board,
            "channels": self.channels,
            "in_planes": self.in_planes,
            "num_blocks": len(self.tower),
            "policy_size": self.policy_size,
            "head_h1": self.channels * self.board * self.board,
            "head_h2": self.value.fc1.out_features,
            "head": "value_wdl_policy",
            "quantization": quantization,
        }


def build_resnet(board=19, channels=64, blocks=6, policy_size=None, in_planes=1, feature_set="go_planes_v01"):
    return ResnetTower(board=board, channels=channels, blocks=blocks, policy_size=policy_size, in_planes=in_planes, feature_set=feature_set)
