import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP


def rectangle_window(x: torch.Tensor) -> torch.Tensor:
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
        bandwidth = ctx.bandwidth

        x_grad = torch.zeros_like(x)
        rect = rectangle_window((x - threshold) / bandwidth)
        threshold_grad = -(1.0 / bandwidth) * rect * grad_output

        while threshold_grad.ndim > threshold.ndim:
            threshold_grad = threshold_grad.sum(dim=0)

        return x_grad, threshold_grad, None


class JumpReLUSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: torch.Tensor, bandwidth: float):
        ctx.save_for_backward(x, threshold)
        ctx.bandwidth = bandwidth
        return x * (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, threshold = ctx.saved_tensors
        bandwidth = ctx.bandwidth

        x_grad = (x > threshold).to(x.dtype) * grad_output

        rect = rectangle_window((x - threshold) / bandwidth)
        threshold_grad = -(threshold / bandwidth) * rect * grad_output

        while threshold_grad.ndim > threshold.ndim:
            threshold_grad = threshold_grad.sum(dim=0)

        return x_grad, threshold_grad, None


def step_ste(x: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
    return StepSTE.apply(x, threshold, bandwidth)


def jumprelu_ste(x: torch.Tensor, threshold: torch.Tensor, bandwidth: float) -> torch.Tensor:
    return JumpReLUSTE.apply(x, threshold, bandwidth)


class TinyJumpReLUSAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        latent_factor: int = 4,
        input_scale: float = 1.0,
        init_threshold: float = 1e-3,
        ste_bandwidth: float = 1e-3,
    ):
        super().__init__()
        d_latent = d_in * latent_factor

        self.d_in = d_in
        self.d_latent = d_latent
        self.ste_bandwidth = ste_bandwidth

        self.register_buffer(
            "input_scale",
            torch.tensor(float(input_scale), dtype=torch.float32),
        )

        self.W_enc = nn.Parameter(torch.empty(d_latent, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_latent))

        self.W_dec = nn.Parameter(torch.empty(d_in, d_latent))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        init_log_threshold = math.log(init_threshold)
        self.log_threshold = nn.Parameter(
            torch.full((d_latent,), init_log_threshold, dtype=torch.float32)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self._normalise_decoder_()

        with torch.no_grad():
            self.W_enc.copy_(self.W_dec.T)

    @torch.no_grad()
    def _normalise_decoder_(self) -> None:
        norms = self.W_dec.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    @torch.no_grad()
    def remove_decoder_grad_parallel_(self) -> None:
        if self.W_dec.grad is None:
            return

        col_norms = self.W_dec.data.norm(dim=0, keepdim=True).clamp(min=1e-8)
        unit_cols = self.W_dec.data / col_norms
        parallel = (self.W_dec.grad * unit_cols).sum(dim=0, keepdim=True) * unit_cols
        self.W_dec.grad.sub_(parallel)

    def threshold(self) -> torch.Tensor:
        return torch.exp(self.log_threshold)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_scaled = x / self.input_scale
        x_in = x_scaled - self.b_dec

        pre_acts = F.relu(x_in @ self.W_enc.T + self.b_enc)
        theta = self.threshold()
        z = jumprelu_ste(pre_acts, theta, self.ste_bandwidth)
        x_hat = z @ self.W_dec.T + self.b_dec

        return x_hat, z, pre_acts, x_scaled


def module_of(model: nn.Module) -> nn.Module:
    if isinstance(model, DDP):
        return model.module
    return model


def compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(torch.sum(p.grad.detach() ** 2).item())
    return total ** 0.5