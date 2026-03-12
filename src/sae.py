import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from config import INIT_THRESHOLD, LATENT_DIM, LATENT_FACTOR, STE_BANDWIDTH


def rectangle_window(x: torch.Tensor):
    return ((x > -0.5) & (x < 0.5)).to(x.dtype)


class StepSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors
        rect = rectangle_window((x - threshold) / ctx.bandwidth)
        threshold_grad = -(1.0 / ctx.bandwidth) * rect * grad_output
        while threshold_grad.ndim > threshold.ndim:
            threshold_grad = threshold_grad.sum(dim=0)
        return torch.zeros_like(x), threshold_grad, None


class JumpReLUSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors
        x_grad = (x > threshold).to(x.dtype) * grad_output
        rect = rectangle_window((x - threshold) / ctx.bandwidth)
        threshold_grad = -(threshold / ctx.bandwidth) * rect * grad_output
        while threshold_grad.ndim > threshold.ndim:
            threshold_grad = threshold_grad.sum(dim=0)
        return x_grad, threshold_grad, None


def step_ste(x: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
    return StepSTE.apply(x, threshold, bandwidth)


def jumprelu_ste(x: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
    return JumpReLUSTE.apply(x, threshold, bandwidth)


class TinyJumpReLUSAE(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        if LATENT_DIM is not None:
            d_latent = int(LATENT_DIM)
        elif LATENT_FACTOR is not None:
            d_latent = int(d_in * LATENT_FACTOR)
        else:
            raise ValueError("Set either LATENT_DIM or LATENT_FACTOR in config.py")

        self.d_in = d_in
        self.d_latent = d_latent

        self.W_enc = nn.Parameter(torch.empty(d_latent, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_latent))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_latent))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.log_threshold = nn.Parameter(
            torch.full((d_latent,), math.log(INIT_THRESHOLD), dtype=torch.float32)
        )

        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self.normalise_decoder()

        with torch.no_grad():
            self.W_enc.copy_(self.W_dec.T)

    def threshold(self):
        return torch.exp(self.log_threshold)

    @torch.no_grad()
    def normalise_decoder(self):
        self.W_dec.div_(self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8))

    @torch.no_grad()
    def remove_decoder_grad_parallel(self):
        if self.W_dec.grad is None:
            return
        unit = self.W_dec / self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.grad.sub_((self.W_dec.grad * unit).sum(dim=0, keepdim=True) * unit)

    def forward(self, x: torch.Tensor):
        x_in = x - self.b_dec
        pre = F.relu(x_in @ self.W_enc.T + self.b_enc)
        z = jumprelu_ste(pre, self.threshold(), STE_BANDWIDTH)
        x_hat = z @ self.W_dec.T + self.b_dec
        return x_hat, pre


def module_of(model: nn.Module):
    if isinstance(model, DDP):
        return model.module
    return model