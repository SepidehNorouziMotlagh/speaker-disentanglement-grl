"""Gradient Reversal Layer (Ganin & Lempitsky, 2015).

Forward pass is identity. Backward pass multiplies the incoming gradient
by -lambda. Placed between a shared representation and a domain/speaker
classifier to encourage the representation to become invariant to the
classified attribute.
"""

from __future__ import annotations

import torch
from torch.autograd import Function


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(torch.nn.Module):
    def __init__(self, lambda_: float = 1.0) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)