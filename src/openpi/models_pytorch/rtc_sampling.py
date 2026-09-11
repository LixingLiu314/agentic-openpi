"""Inference-time RTC (Black et al., 2025), in pi05's noise=1 time convention.

No parameters or training recipes are changed. Guidance differentiates the
denoised endpoint with respect to the noisy actions, never accumulating weight
gradients. The overlap is expressed in the CURRENT observation's model space.
"""
import math

import torch


def overlap_weights(horizon, overlap, delay, *, device=None):
    if not 0 < overlap < horizon or not 0 <= delay <= overlap:
        raise ValueError("RTC requires a valid remaining overlap and delay")
    i = torch.arange(horizon, dtype=torch.float32, device=device)
    c = ((overlap - i) / (overlap - delay + 1)).clamp(0, 1)
    soft = c * torch.expm1(c) / math.expm1(1.0)
    return torch.where(i < delay, 1.0, torch.where(i < overlap, soft, 0.0))


def guided_flow(velocity_fn, noise, target, weights, *, num_steps=10, max_guidance=5.0):
    """Euler integration with the paper's endpoint VJP / soft inpainting mask."""
    if num_steps < 1 or not 0 < max_guidance <= 10:
        raise ValueError("Invalid RTC sampler settings")
    if noise.shape != target.shape or weights.shape != noise.shape:
        raise ValueError("RTC tensors must have matching [B,H,D] shapes")
    if not all(torch.isfinite(x).all() for x in (noise, target, weights)):
        raise ValueError("Nonfinite RTC sampler input")
    if (weights < 0).any() or (weights > 1).any():
        raise ValueError("RTC weights must lie in [0,1]")
    x = noise.detach()
    target, weights = target.detach(), weights.detach()
    dt = -1.0 / num_steps
    for step in range(num_steps):
        t = 1.0 - step / num_steps
        with torch.enable_grad():
            noisy = x.detach().requires_grad_(True)
            velocity = velocity_fn(noisy, torch.full((x.shape[0],), t, device=x.device))
            endpoint = noisy - t * velocity
            error = ((target - endpoint) * weights).detach()
            correction = torch.autograd.grad(endpoint, noisy, grad_outputs=error,
                                             retain_graph=False, create_graph=False)[0]
        # Paper tau = 1 - t. Its positive data-direction correction subtracts
        # from pi05's reverse-time velocity (which points from data to noise).
        coefficient = min(max_guidance, (t*t + (1-t)**2) / max(t*(1-t), 1e-8))
        x = (noisy.detach() + dt * (velocity.detach() - coefficient * correction.detach())).detach()
        if not torch.isfinite(x).all():
            raise FloatingPointError("RTC sampler produced nonfinite actions")
    return x


def sample_rtc(model, context, prefix, previous, delay, *, noise=None, num_steps=10):
    horizon, dim = model.base.config.action_horizon, model.base.config.action_dim
    device = context.state.device
    if previous.ndim != 3 or previous.shape[0] != 1 or previous.shape[2] != 14:
        raise ValueError("RTC prior must be one native14 overlap in model coordinates")
    target = torch.zeros((1, horizon, dim), device=device, dtype=torch.float32)
    target[:, :previous.shape[1], :14] = previous
    weights = torch.zeros_like(target)
    weights[:, :, :14] = overlap_weights(horizon, previous.shape[1], delay, device=device)[None, :, None]
    if noise is None:
        noise = model.base.sample_noise(target.shape, device)
    noise = torch.as_tensor(noise, device=device, dtype=torch.float32)
    if noise.ndim == 2:
        noise = noise[None]
    return guided_flow(lambda x, t: model.base.denoise_step(context.state, prefix.mask, prefix.cache, x, t),
                       noise, target, weights, num_steps=num_steps)
