import torch
import torch.nn as nn
from .util import RoundSTE


class UniformQuantizer(nn.Module):
    """
    Uniform symmetric quantizer with max/min scale factor
    """
    def __init__(self, bit_width: int):
        super(UniformQuantizer, self).__init__()
        self.bit_width = bit_width

    def set_bits(self, new_bit_width: int) -> None:
        self.bit_width = new_bit_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        max_val = torch.max(torch.abs(x))
        s = max_val / (2 ** (self.bit_width - 1) - 1)

        return RoundSTE.apply(x / s) * s
