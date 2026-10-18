"""Flash Next's NVFP4 MoE: routed experts on the grouped NVFP4 kernel, the bf16 shared expert on the FP4 tables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch

from tensorfold.cuda.nvfp4 import experts as nvx

from . import nvfp4

TILE_ENV = "TF_FLASHNEXT_PROMPT_EXPERT_TILE"
TILE_MODES = {"0": 0, "1": 1, "2": 2}     # off; gate/up only; gate/up and down


def tile_mode() -> int:
    """TF_FLASHNEXT_PROMPT_EXPERT_TILE: which faces of prompt chunks run the shared-memory tile (unset: 2, both)."""

    value = os.environ.get(TILE_ENV, "").strip()
    if not value:
        return 2
    if value not in TILE_MODES:
        raise ValueError(f"{TILE_ENV}: 0 (off), 1 (gate/up) or 2 (gate/up and down), not {value!r}")
    return TILE_MODES[value]


SHARED_DECODE_ENV = "TF_FP4_SHARED_DECODE"
SHARED_DECODE_MODES = ("qmmf", "prompt")


def shared_decode() -> str:
    """TF_FP4_SHARED_DECODE: the kernel of a decode window's MXFP8 shared expert (unset: qmmf).

    Decode windows keep bf16 slots (``moe_prefill``), which once sent their shared expert to the prompt GEMM. ``qmmf``
    runs it on the lane matmul like every other decode linear; ``prompt`` keeps the prompt GEMM's bits. Prompt rows
    take the prompt GEMM either way."""

    value = os.environ.get(SHARED_DECODE_ENV, "").strip()
    if not value:
        return "qmmf"
    if value not in SHARED_DECODE_MODES:
        raise ValueError(f"{SHARED_DECODE_ENV}: qmmf (the lane matmul) or prompt (the prompt GEMM), not {value!r}")
    return value


def tile_faces(mode: int, rows: int, prompt: bool) -> tuple[bool, bool]:
    """(gate/up, down) on the tile kernel: prompt buffers of at least ``nvx.TILE_ROWS`` rows, as ``mode`` allows."""

    on = prompt and rows >= nvx.TILE_ROWS
    return on and mode >= 1, on and mode == 2


@dataclass
class Expert4:
    """One expert: gate/up rows in one grid (gate rows first), and the down grid."""

    gu: nvfp4.FP4         # [2*NI, K]
    down: nvfp4.FP4       # [K, NI]


@dataclass
class MoE4:
    """One layer's experts: the routed ones as grouped NVFP4 blocks, the shared one as bf16 tables."""

    routed_experts: nvx.Experts4
    shared: Expert4
    kernel: str = "nvfp4"
    prompt_tile: int = field(default_factory=tile_mode)    # read once, when the layer loads
    shared_decode: str = field(default_factory=shared_decode)

    capturable = True     # the plan stays on the device

    @property
    def routed(self) -> int:
        return self.routed_experts.count

    @property
    def count(self) -> int:
        return self.routed + 1

    @property
    def width(self) -> int:
        return self.routed_experts.width                     # NI: a gate row's inputs, down's outputs

    @property
    def dims(self) -> int:
        return self.routed_experts.dims                      # D: x's inputs, down's outputs

    def nbytes(self) -> int:
        return self.routed_experts.nbytes() + self.shared.gu.nbytes() + self.shared.down.nbytes()

    def shared_act(self, x: torch.Tensor, prompt: bool = False) -> torch.Tensor:
        """x [R, D] bf16 -> the shared expert's SwiGLU [R, NI] bf16, bf16(bf16(silu(g)) * u) on bf16 g and u."""

        g = _shared(self.shared.gu, x, prompt, False)
        gate, up = g[:, :self.width].to(torch.float32), g[:, self.width:].to(torch.float32)
        return ((gate / (1.0 + torch.exp(-gate))).to(torch.bfloat16).to(torch.float32) * up).to(torch.bfloat16)


def _shared(lin, x: torch.Tensor, prompt: bool, f32: bool) -> torch.Tensor:
    """The shared expert's matmul: bf16 weights on the FP4 kernel's exact tables, MXFP8 ones on the lane matmul."""

    if isinstance(lin, nvfp4.FP4):
        return nvfp4.matmul(x, lin, f32=f32)
    return lin.prefill(x) if prompt else lin(x)


def expert4_from_bf16(gate: torch.Tensor, up: torch.Tensor, down: torch.Tensor) -> Expert4:
    """The shared expert from bf16 weights (torch linear layout [out, in]): identity-scaled FP4 tables."""

    gu = nvfp4.fp4_from_bf16(torch.cat([gate, up], dim=0).contiguous())
    return Expert4(gu, nvfp4.fp4_from_bf16(down.contiguous()))


def moe4_from_checkpoint(gate: tuple, up: tuple, down: tuple,
                         shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> MoE4:
    """A layer from stacked checkpoint arrays (words, e4m3 scales, per-expert scales) and the bf16 shared expert."""

    return MoE4(nvx.make(gate, up, down), shared if isinstance(shared, Expert4) else expert4_from_bf16(*shared))


def moe4_from_bf16(gate_up: torch.Tensor, down: torch.Tensor,
                   shared: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> MoE4:
    """The MTP layer's bf16 experts (gate_up [E, 2NI, K], down [E, K, NI]) in NVFP4: they only draft."""

    ni = gate_up.shape[1] // 2
    gate, up = nvx.quantize(gate_up[:, :ni]), nvx.quantize(gate_up[:, ni:])
    return MoE4(nvx.make(gate, up, nvx.quantize(down)),
                shared if isinstance(shared, Expert4) else expert4_from_bf16(*shared))


def moe4_from_experts(gate: list, up: list, down: list, shared: tuple) -> MoE4:
    """Per-expert FP4 triples and the bf16 shared expert -> a layer (the test path)."""

    def stack(items):
        return (torch.stack([t[0] for t in items]), torch.stack([t[1].view(torch.uint8) for t in items]),
                torch.stack([torch.as_tensor(t[2], dtype=torch.float32).reshape(()) for t in items]).to(
                    items[0][0].device))

    return moe4_from_checkpoint(stack(gate), stack(up), stack(down), shared)


def moe(x: torch.Tensor, xs: torch.Tensor, router_rows: torch.Tensor, ex: MoE4, buf, cfg, *,
        decode: bool = False) -> None:
    """Route rows x [R, D] and run their experts into buf (the shared ``moe`` contract); ``xs`` unused: no biases.

    ``decode``: a decode window's rows, whose bf16 slots alone do not tell them from a prompt's."""

    from tensorfold.cuda import experts as grouped
    from tensorfold.cuda import moe as moe_mod

    rows, top_k = x.shape[0], int(cfg.num_experts_per_tok)
    prompt = buf.y.dtype != torch.float32                  # a prompt's buffers keep bf16 slots
    gu_tile, dn_tile = tile_faces(ex.prompt_tile, rows, prompt)
    shared_prompt = prompt and not (decode and ex.shared_decode == "qmmf")
    moe_mod.router(x, router_rows, buf.logits[:rows])
    moe_mod.select(buf.logits[:rows], buf, top_k, ex.routed, nvx.TILE_ITEM if gu_tile else nvx.PREFILL_TILE)
    nvx.gate_up(x, ex.routed_experts, buf.plan, buf.act.view(-1, ex.width), rows, skip=ex.routed, tile=gu_tile)
    buf.act[:rows, top_k] = ex.shared_act(x, shared_prompt)
    if dn_tile != gu_tile:                                 # mode 1: down regroups in its own kernel's items
        grouped.route(buf.pick[:rows], buf.plan, nvx.TILE_ITEM if dn_tile else nvx.PREFILL_TILE)
    nvx.down(buf.act.view(-1, ex.width), ex.routed_experts, buf.plan, buf.y.view(-1, ex.dims), rows, skip=ex.routed,
             tile=dn_tile)
    buf.y[:rows, top_k] = _shared(ex.shared.down, buf.act[:rows, top_k], shared_prompt, not prompt)
    return buf
