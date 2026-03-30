from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from vllm import _custom_ops as vllm_ops
import vllm.model_executor.layers.activation


State = dict[str, object]
SetupFn = Callable[[tuple[int, int], torch.dtype], State]
RunFn = Callable[[State], tuple[torch.Tensor, ...]]
BytesFn = Callable[[tuple[int, int]], int]


@dataclass(frozen=True)
class KernelSpec:
    setup: SetupFn
    reference: RunFn
    implementations: list[tuple[str, RunFn]]
    bytes_processed: BytesFn


def clone_state(state: State) -> State:
    cloned: State = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            cloned[key] = value.clone()
        else:
            cloned[key] = value
    return cloned


def silu_setup(shape: tuple[int, int], dtype: torch.dtype) -> State:
    batch, hidden = shape
    return {
        "x": torch.randn((batch, hidden * 2), device="cuda", dtype=dtype),
        "hidden": hidden,
    }


def silu_reference(state: State) -> tuple[torch.Tensor, ...]:
    x = state["x"]
    hidden = x.shape[-1] // 2
    return (F.silu(x[..., :hidden]) * x[..., hidden:],)


def silu_vllm(state: State) -> tuple[torch.Tensor, ...]:
    x = state["x"]
    hidden = int(state["hidden"])
    out = torch.empty((x.shape[0], hidden), device=x.device, dtype=x.dtype)
    torch.ops._C.silu_and_mul(out, x)
    return (out,)


def build_silu_torch_compile() -> RunFn:
    @torch.compile(dynamic=False)
    def compiled(x: torch.Tensor) -> torch.Tensor:
        hidden = x.shape[-1] // 2
        return F.silu(x[..., :hidden]) * x[..., hidden:]

    return lambda state: (compiled(state["x"]),)


def silu_torch_eager(state: State) -> tuple[torch.Tensor, ...]:
    x = state["x"]
    hidden = x.shape[-1] // 2
    return (F.silu(x[..., :hidden]) * x[..., hidden:],)


def fused_add_rms_norm_setup(shape: tuple[int, int], dtype: torch.dtype) -> State:
    batch, hidden = shape
    return {
        "x": torch.randn((batch, hidden), device="cuda", dtype=dtype),
        "weight": torch.randn((hidden,), device="cuda", dtype=dtype),
        "eps": 1e-6,
    }


def fused_add_rms_norm_reference(state: State) -> tuple[torch.Tensor, ...]:
    x = state["x"]
    weight = state["weight"]
    eps = state["eps"]
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    normalized = x * torch.rsqrt(variance + eps)
    return ((normalized.to(x.dtype) * weight),)


def fused_add_rms_norm_vllm(state: State) -> tuple[torch.Tensor, ...]:
    out = torch.empty_like(state["x"])
    vllm_ops.rms_norm(out, state["x"], state["weight"], state["eps"])
    return (out,)


def build_fused_add_rms_norm_torch_compile() -> RunFn:
    @torch.compile(dynamic=False)
    def compiled(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        return x_normed.to(x.dtype) * weight

    return lambda state: (
        compiled(
            state["x"],
            state["weight"],
            state["eps"],
        ),
    )


def fused_add_rms_norm_torch_eager(state: State) -> tuple[torch.Tensor, ...]:
    x = state["x"]
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + state["eps"])
    return ((x_normed.to(x.dtype) * state["weight"]),)


def build_cos_sin_cache(max_position: int, head_size: int, dtype: torch.dtype) -> torch.Tensor:
    theta = torch.arange(0, head_size, 2, device="cuda", dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (theta / head_size))
    freqs = torch.outer(
        torch.arange(max_position, device="cuda", dtype=torch.float32),
        inv_freq,
    )
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


def rotary_setup(shape: tuple[int, int], dtype: torch.dtype) -> State:
    tokens, hidden = shape
    head_size = 64
    if hidden % head_size != 0:
        raise ValueError(
            f"rotary_embedding requires hidden size divisible by {head_size}, got {hidden}"
        )
    num_heads = hidden // head_size
    positions = torch.arange(tokens, device="cuda", dtype=torch.long).unsqueeze(0)
    return {
        "positions": positions,
        "q": torch.randn((1, tokens, num_heads, head_size), device="cuda", dtype=dtype),
        "k": torch.randn((1, tokens, num_heads, head_size), device="cuda", dtype=dtype),
        "cache": build_cos_sin_cache(max(2048, tokens), head_size, dtype),
        "head_size": head_size,
        "is_neox": True,
    }


def rotary_reference(state: State) -> tuple[torch.Tensor, ...]:
    positions = state["positions"]
    q = state["q"]
    k = state["k"]
    cache = state["cache"]
    half = q.shape[-1] // 2
    selected = cache.index_select(0, positions.reshape(-1))
    cos = selected[:, :half].view(1, positions.shape[1], 1, half)
    sin = selected[:, half:].view(1, positions.shape[1], 1, half)
    q0, q1 = q[..., :half], q[..., half:]
    k0, k1 = k[..., :half], k[..., half:]
    q.copy_(torch.cat((q0 * cos - q1 * sin, q1 * cos + q0 * sin), dim=-1))
    k.copy_(torch.cat((k0 * cos - k1 * sin, k1 * cos + k0 * sin), dim=-1))
    return (q, k)


def rotary_vllm(state: State) -> tuple[torch.Tensor, ...]:
    vllm_ops.rotary_embedding(
        state["positions"],
        state["q"],
        state["k"],
        state["head_size"],
        state["cache"],
        state["is_neox"],
    )
    return (state["q"], state["k"])


def build_rotary_torch_compile() -> RunFn:
    @torch.compile(dynamic=False)
    def compiled(
        positions: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        half = q.shape[-1] // 2
        selected = cache.index_select(0, positions.reshape(-1))
        cos = selected[:, :half].view(1, positions.shape[1], 1, half)
        sin = selected[:, half:].view(1, positions.shape[1], 1, half)
        q0, q1 = q[..., :half], q[..., half:]
        k0, k1 = k[..., :half], k[..., half:]
        q.copy_(torch.cat((q0 * cos - q1 * sin, q1 * cos + q0 * sin), dim=-1))
        k.copy_(torch.cat((k0 * cos - k1 * sin, k1 * cos + k0 * sin), dim=-1))
        return q, k

    return lambda state: compiled(
        state["positions"],
        state["q"],
        state["k"],
        state["cache"],
    )


def rotary_torch_eager(state: State) -> tuple[torch.Tensor, ...]:
    return rotary_reference(state)


def build_registry(bytes_per_element: int) -> dict[str, KernelSpec]:
    return {
        "silu_and_mul": KernelSpec(
            setup=silu_setup,
            reference=silu_reference,
            implementations=[
                ("vllm-cuda", silu_vllm),
                ("torch-compile", build_silu_torch_compile()),
                ("torch-eager", silu_torch_eager),
            ],
            bytes_processed=lambda shape: shape[0] * shape[1] * 3 * bytes_per_element,
        ),
        "fused_add_rms_norm": KernelSpec(
            setup=fused_add_rms_norm_setup,
            reference=fused_add_rms_norm_reference,
            implementations=[
                ("vllm-cuda", fused_add_rms_norm_vllm),
                ("torch-compile", build_fused_add_rms_norm_torch_compile()),
                ("torch-eager", fused_add_rms_norm_torch_eager),
            ],
            bytes_processed=lambda shape: shape[0] * shape[1] * 3 * bytes_per_element,
        ),
        "rotary_embedding": KernelSpec(
            setup=rotary_setup,
            reference=rotary_reference,
            implementations=[
                ("vllm-cuda", rotary_vllm),
                ("torch-compile", build_rotary_torch_compile()),
                ("torch-eager", rotary_torch_eager),
            ],
            bytes_processed=lambda shape: shape[0] * shape[1] * 4 * bytes_per_element,
        ),
    }
