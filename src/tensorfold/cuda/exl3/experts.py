"""Routed EXL3 experts of any codebook, a width per expert, one grouped launch a projection; rows never depend on the window."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import torch

from . import switches

CB_3INST, CB_MCG, CB_MUL1 = 0, 1, 2
ACT_BF16, ACT_F32 = 0, 1          # SwiGLU with the GLM family's bf16 roundings / in fp32
# Rate codes with a kernel instance. Half-bits a value: 1..8 bits (2, 4, .. 16) and every half-integer rate
# 1.5..7.5 (3, 5, .. 15). A rate between those is RATE_WORDS + the tile's 32-bit words: 2.25 bits, 18 words, is 118.
RATE_WORDS = 100
K2_SUPPORTED = (*range(2, 17), RATE_WORDS + 18)

# (n tiles a block, warps, K splits, tiles in flight): GLM's settings, whose arithmetic order this keeps bit for bit
GLM_GATEUP = (8, 4, 4, 1)
GLM_DOWN = (8, 4, 1, 1)
PROMPT_ROWS = 64          # a window of more rows than this is a prompt's: its experts each serve many rows
PROMPT_TILES = 2          # row tiles (16 rows each) that share one read of an expert's weights in such a window
DECODE_ROWS = 16          # a window of at most this many rows fits one row tile: decode_kernel (TF_EXL3_EXPERT_DECODE)
# decode_kernel's (n tiles a block, k steps in flight a warp) by projection shape (K, N) and window rows: entries
# (most rows, tile), the first whose rows cover the window; None keeps grouped_kernel. Every tile writes the same bytes.
# GB10 sweep (tools/exl3_expert_sweep.py, tf-runs/gpu-exl3dec/RESULTS.md, 2026-10-19), against grouped_kernel:
# Flash Next gate/up 4x4 1.09x at 1 row, 4x2 1.01-1.10x at 2-5 rows and 0.99-1.00x from 6; Flash Next down 4x4
# 1.06-1.09x at 1-4 rows and 0.98-1.00x from 5, where 4x2 is 1.00-1.04x; GLM-5.3 4x2 1.07-1.23x at every row count,
# 4x4 0.93-1.07x. 2x4 and 2x8 never won.
DECODE_TILES: dict[tuple[int, int], tuple[tuple[int, tuple[int, int] | None], ...]] = {
    (2560, 640): ((1, (4, 4)), (5, (4, 2)), (16, None)),       # Flash Next gate/up
    (640, 2560): ((4, (4, 4)), (16, (4, 2))),                  # Flash Next down
    (4096, 2048): ((16, (4, 2)),),                             # GLM-5.3 gate/up
    (2048, 4096): ((16, (4, 2)),),                             # GLM-5.3 down
}
DECODE_TILE = (4, 2)      # another shape: the setting that was at least grouped_kernel's speed (0.99x) everywhere
SCAN_EXPERTS = 1024       # group_scan_kernel (TF_EXL3_GROUP_SCAN): an expert a thread of a 1,024-thread block
SCAN_PICKS = 65535        # and ranks of 16 bits
EPI_GATEUP, EPI_DOWN = 1, 2   # grouped_kernel's fused epilogues (TF_EXL3_GATEUP_FUSE, TF_EXL3_DOWN_FUSE)


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    srcs = [str(here / f) for f in ("experts.cpp", "experts.cu", "experts_cb0.cu", "experts_cb1.cu", "experts_cb2.cu")]
    return load(name="tensorfold_exl3_experts_v7", sources=srcs, extra_cuda_cflags=["-O3", "-lineinfo"],
                verbose=False)


def codebook_id(name: str) -> int:
    """'3inst' / 'mcg' / 'mul1' (the checkpoint's quantization_config.codebook, or the marker tensor's name)."""

    ids = {"3inst": CB_3INST, "mcg": CB_MCG, "mul1": CB_MUL1}
    if name not in ids:
        raise ValueError(f"unknown EXL3 codebook {name!r}")
    return ids[name]


def k2_of(trellis: torch.Tensor) -> int:
    """The rate code of a trellis int16 [K/16, N/16, 16 * bits]: half-bits a value, or RATE_WORDS + words a tile."""

    w = trellis.shape[-1]
    if w % 2:
        raise ValueError(f"trellis last dim {w} is not a whole count of 32-bit words")
    k2 = w // 8 if w % 8 == 0 else RATE_WORDS + w // 2
    if k2 not in K2_SUPPORTED:
        raise ValueError(f"no kernel reads the EXL3 rate of {w / 16:g} bits")
    return k2


def tile_words(k2: int) -> int:
    """32-bit words a tile at a rate code."""

    return k2 - RATE_WORDS if k2 > RATE_WORDS else 4 * k2


def check_window(window: int) -> int:
    if not 8 <= int(window) <= 16:
        raise ValueError(f"an EXL3 codeword window is 8 to 16 bits, got {window}")
    return int(window)


@dataclass
class Exl3RoutedExperts:
    """One layer's routed experts: trellis pointers and widths per projection, stacked suh/svh, the tensors keeping the trellises alive."""

    gate_ptr: torch.Tensor    # int64 [E]
    up_ptr: torch.Tensor
    down_ptr: torch.Tensor
    gate_k2: torch.Tensor     # int32 [E], rate codes (K2_SUPPORTED)
    up_k2: torch.Tensor
    down_k2: torch.Tensor
    suh_g: torch.Tensor       # fp16 [E, D]
    suh_u: torch.Tensor
    svh_g: torch.Tensor       # fp16 [E, I]
    svh_u: torch.Tensor
    suh_d: torch.Tensor       # fp16 [E, I]
    svh_d: torch.Tensor       # fp16 [E, D]
    count: int                # E
    dims: int                 # D (model width)
    width: int                # I (expert width on this rank)
    cb: int
    k2_gu: tuple[int, int]    # (min, max) K2 over gate and up
    k2_d: tuple[int, int]
    trellis_bytes: torch.Tensor   # int64 [E], gate + up + down trellis bytes of each expert (for GB/s)
    keep: list = field(default_factory=list, repr=False)
    window: int = 16          # the bits of a 16-bit window that make the codeword (a sushi pack names fewer)
    aligned: bool = True      # every trellis starts on 16 bytes (the staged path's 16-byte copies need it)

    def nbytes_read(self, ids: Sequence[int]) -> int:
        return int(self.trellis_bytes[list(ids)].sum())


def prepare(gate: Sequence[tuple], up: Sequence[tuple], down: Sequence[tuple], codebook: int | str,
            device="cuda", window: int = 16) -> Exl3RoutedExperts:
    """A layer from per-expert (trellis, suh, svh) triples, trellises referenced in place, each at its own width."""

    cb = codebook_id(codebook) if isinstance(codebook, str) else int(codebook)
    window = check_window(window)
    E = len(gate)
    if not (len(up) == len(down) == E) or E == 0:
        raise ValueError("gate, up and down need the same, non-zero number of experts")
    D, I = gate[0][0].shape[0] * 16, gate[0][0].shape[1] * 16
    keep = []

    def table(mats, k, n):
        ptrs, k2s = [], []
        for t, _, _ in mats:
            if t.dtype != torch.int16 or not t.is_contiguous() or t.device.type != "cuda":
                raise ValueError("trellis must be a contiguous CUDA int16 tensor")
            if t.shape[0] * 16 != k or t.shape[1] * 16 != n:
                raise ValueError(f"expert shape {tuple(t.shape)} does not match [{k // 16}, {n // 16}, *]")
            k2s.append(k2_of(t))
            ptrs.append(t.data_ptr())
            keep.append(t)
        return (torch.tensor(ptrs, dtype=torch.int64, device=device),
                torch.tensor(k2s, dtype=torch.int32, device=device), k2s)

    gp, gk, gks = table(gate, D, I)
    upp, uk, uks = table(up, D, I)
    dp, dk, dks = table(down, I, D)
    for k2s in (gks + uks, dks):            # the kernel's own check, here at load rather than at the first forward
        if max(k2s) > 16 and min(k2s) != max(k2s):
            raise ValueError("a rate between ExLlamaV3's widths must be its layer's only rate")

    def stack(mats, j, n):
        out = torch.empty((E, n), dtype=torch.float16, device=device)
        for e, m in enumerate(mats):
            out[e].copy_(m[j].reshape(-1))
        return out

    tb = torch.tensor([(D * I // 256) * sum(tile_words(k) for k in (gks[e], uks[e], dks[e])) * 4
                       for e in range(E)], dtype=torch.int64)
    return Exl3RoutedExperts(gp, upp, dp, gk, uk, dk, stack(gate, 1, D), stack(up, 1, D), stack(gate, 2, I),
                             stack(up, 2, I), stack(down, 1, I), stack(down, 2, D), E, D, I, cb,
                             (min(gks + uks), max(gks + uks)), (min(dks), max(dks)), tb, keep, window,
                             all(t.data_ptr() % 16 == 0 for t in keep))


def prepare_stacked(gt: torch.Tensor, ut: torch.Tensor, dt: torch.Tensor, suh_g, suh_u, svh_g, svh_u, suh_d, svh_d,
                    codebook: int | str, window: int = 16) -> Exl3RoutedExperts:
    """A uniform-width layer stacked per projection: trellis [E, K/16, N/16, 16K] (int16, sushi's uint16, or GLM's int32 words), suh/svh [E, n]."""

    def as16(t):
        return t.view(torch.int16) if t.dtype in (torch.int32, torch.uint16) else t

    gt, ut, dt = as16(gt), as16(ut), as16(dt)
    dev, E = gt.device, gt.shape[0]
    D, I = gt.shape[1] * 16, gt.shape[2] * 16
    if tuple(ut.shape[:3]) != (E, D // 16, I // 16) or tuple(dt.shape[:3]) != (E, I // 16, D // 16):
        raise ValueError("stacked gate, up and down trellises disagree on experts or shape")

    def table(t):
        # the stack is read where it is: expert e's tiles start e slices into it
        if t.dtype != torch.int16 or not t.is_contiguous() or t.device.type != "cuda":
            raise ValueError("a stacked trellis must be a contiguous CUDA tensor of 16-bit words")
        k2 = k2_of(t)
        ptrs = t.data_ptr() + t[0].numel() * 2 * torch.arange(E, dtype=torch.int64)
        return ptrs.to(dev), torch.full((E,), k2, dtype=torch.int32, device=dev), k2

    def scales(s, n):
        return s.to(device=dev, dtype=torch.float16).reshape(E, n).contiguous()

    (gp, gk, g2), (upp, uk, u2), (dp, dk, d2) = table(gt), table(ut), table(dt)
    tb = torch.full((E,), (D * I // 256) * sum(tile_words(k) for k in (g2, u2, d2)) * 4, dtype=torch.int64)
    cb = codebook_id(codebook) if isinstance(codebook, str) else int(codebook)
    aligned = all(t.data_ptr() % 16 == 0 and t[0].numel() * 2 % 16 == 0 for t in (gt, ut, dt))
    return Exl3RoutedExperts(gp, upp, dp, gk, uk, dk, scales(suh_g, D), scales(suh_u, D), scales(svh_g, I),
                             scales(svh_u, I), scales(suh_d, I), scales(svh_d, D), E, D, I, cb,
                             (min(g2, u2), max(g2, u2)), (d2, d2), tb, [gt, ut, dt], check_window(window), aligned)


def default_config(K: int, N: int, gateup: bool) -> tuple[int, int, int, int]:
    """The tile setting for a K -> N projection (GLM's where it divides): the shape's alone, so rows stay independent."""

    cands = [GLM_GATEUP if gateup else GLM_DOWN, (8, 4, 2, 1), (8, 4, 1, 1), (4, 4, 2, 2), (4, 4, 1, 2)]
    for nt, w, sk, pf in cands:
        if K % (16 * sk * w) == 0 and N % (16 * nt) == 0:
            return nt, w, sk, pf
    raise ValueError(f"no tile setting divides K={K}, N={N}")


class Scratch:
    """Buffers for up to ``rows`` rows of ``slots`` slots; slots whose pick is not a routed expert are left to the caller."""

    def __init__(self, ex: Exl3RoutedExperts, rows: int, slots: int, cfg_gu=None, cfg_d=None, device="cuda") -> None:
        D, I = ex.dims, ex.width
        self.cfg_gu = cfg_gu or default_config(D, I, True)
        self.cfg_d = cfg_d or default_config(I, D, False)
        P = rows * slots
        self.xg = torch.zeros((P, D), dtype=torch.float16, device=device)
        self.xu = torch.zeros((P, D), dtype=torch.float16, device=device)
        self.xd = torch.zeros((P, I), dtype=torch.float16, device=device)
        # gate and up write 2 * splits * P * I partials, down splits * P * D
        self.z = torch.zeros((max(2 * self.cfg_gu[2] * I, self.cfg_d[2] * D) * P,), dtype=torch.float32, device=device)
        self.y = torch.zeros((P, D), dtype=torch.float32, device=device)
        maxu = min(P, ex.count)
        self.ids = torch.zeros((maxu,), dtype=torch.int32, device=device)
        self.count = torch.zeros((1,), dtype=torch.int32, device=device)
        self.members_buf = torch.full((maxu * rows,), -1, dtype=torch.int32, device=device)
        # grouped_kernel's listed layout (TF_EXL3_EXPERT_ITEMS): the member groups group_kernel lists, and their count
        self.items_buf = torch.zeros((item_bound(P, maxu, 16),), dtype=torch.int32, device=device)
        self.icount = torch.zeros((1,), dtype=torch.int32, device=device)
        # the fused gate/up epilogue's arrivals, an (item, 128-column n block); each launch leaves them zero
        self.arrivals = torch.zeros((self.items_buf.numel() * max(1, I // 128),), dtype=torch.int32, device=device)
        # TF_EXL3_ROT_FUSE's counters, an (item, matrix) of the gate/up launch; each launch leaves them zero
        self.rot_arrivals = torch.zeros((self.items_buf.numel() * 2,), dtype=torch.int32, device=device)
        self.rows, self.slots, self.count_experts = rows, slots, ex.count

    def window(self, R: int):
        """(ids, members) sized for R rows: the grids only span what R rows can use."""

        maxu = min(R * self.slots, self.count_experts)
        return self.ids[:maxu], self.members_buf[:maxu * R].view(maxu, R)

    def items(self, R: int, tile: int) -> torch.Tensor:
        """The item list sized for R rows in member groups of ``tile`` rows: its length is the listed grid's depth."""

        return self.items_buf[:item_bound(R * self.slots, min(R * self.slots, self.count_experts), tile)]


def item_bound(pairs: int, places: int, tile: int) -> int:
    """Most items group_kernel lists for ``pairs`` picks over ``places`` experts in groups of ``tile``: an expert
    with c members lists ceil(c / tile) <= c / tile + 1, so the list holds at most ceil(pairs / tile) + places."""

    return -(-pairs // tile) + places


def prompt_config(cfg: tuple[int, int, int, int]) -> tuple[int, int, int, int, int]:
    """``cfg`` plus the row tiles a block of a many-row window takes. The warps and K splits are ``cfg``'s, so every
    row's sums are too. GB10, 1,024 rows of Flash Next's shape: the two projections 21.0 -> 18.0 ms; four row tiles,
    fewer n tiles or more tiles in flight were all slower."""

    nt, w, sk, pf = cfg
    return (nt, w, sk, pf, PROMPT_TILES) if (nt, w, pf) == (8, 4, 1) else (nt, w, sk, pf, 1)


def decode_tile(K: int, N: int, R: int) -> tuple[int, int] | None:
    """(n tiles, k steps in flight) of decode_kernel for an R-row window of a K -> N projection, or None for
    grouped_kernel (DECODE_TILES, else DECODE_TILE)."""

    for most, tile in DECODE_TILES.get((K, N), ((DECODE_ROWS, DECODE_TILE),)):
        if R <= most:
            return tile
    return None


def decode_config(cfg: tuple[int, int, int, int], K: int, N: int, R: int) -> tuple[int, int, int, int] | None:
    """``cfg`` for decode_kernel: the same warps and K splits, so every sum is ``cfg``'s, with decode_tile's n tiles
    and k steps in flight; None when the table keeps grouped_kernel or the n tiles do not divide N.

    GB10, Flash Next 3.05 bpw: grouped_kernel's 4-warp blocks ran 3 an SM (168 registers, 32.8 KB of shared memory)
    with one k step in flight a warp; decode_kernel's run 4 (4x4: 128 registers) or 5 (4x2: 96) an SM."""

    _, w, sk, _ = cfg
    tile = decode_tile(K, N, R) if R <= DECODE_ROWS else None
    if tile is None or w != 4 or N % (16 * tile[0]):
        return None
    return tile[0], w, sk, tile[1]


def routed(x: torch.Tensor, pick: torch.Tensor, wts: torch.Tensor | None, ex: Exl3RoutedExperts, s: Scratch,
           out: torch.Tensor | None, R: int, limit: float = math.inf, act_mode: int = ACT_F32,
           group: bool = True, prompt: bool | None = None, y: torch.Tensor | None = None) -> torch.Tensor:
    """Routed experts of R rows (picks >= E skipped): Y per slot, or ``out`` = the wts-weighted sum when ``wts``; no host sync.

    ``prompt`` (default: more than PROMPT_ROWS rows) reads each expert once for 32 of its rows; the values are the same.
    With TF_EXL3_EXPERT_ITEMS (and ``group``) a prompt window's grouping lists the member groups grouped_kernel's
    launches run. On a prompt window TF_EXL3_GROUP_SCAN groups on group_scan_kernel, and on the listed layout
    TF_EXL3_GATEUP_FUSE and TF_EXL3_DOWN_FUSE (without ``wts``) run the epilogues inside grouped_kernel's launches,
    and TF_EXL3_EXPERT_STAGED stages its k steps through shared memory, where TF_EXL3_ROT_FUSE has the staged gate/up
    launch write rot_in's rows itself.
    ``y`` (without ``wts``): a contiguous [R * slots, D] that takes the slots in place of the scratch's: bf16 rounds
    them once, as a copy of the fp32 slots into it would round them; fp32 takes them as they are."""

    ext = _ext()
    D, I, E = ex.dims, ex.width, ex.count
    slots = s.slots
    P = R * slots
    if R > s.rows:
        raise ValueError(f"{R} rows but the scratch holds {s.rows}")
    ids, members = s.window(R)
    sw = switches.current()
    pdl = sw.pdl
    many = R > PROMPT_ROWS if prompt is None else bool(prompt)
    gu, dn = plan(s.cfg_gu, s.cfg_d, I, D, R, many, sw.expert_decode)
    # prompt windows only: on decode windows the list cost about 1% decode (GPU 19 Oct, tf-runs/gpu-exl3s2 alone/)
    tile = listed_tile(gu, dn) if group and sw.expert_items and many else 0
    listed = {"items": s.items(R, tile), "icount": s.icount} if tile else {}
    if group:
        # TF_EXL3_GROUP_SCAN on prompt windows, TF_EXL3_GROUP_SCAN_DECODE on decode windows
        scan = {"scan": True} if (sw.group_scan if many else sw.group_scan_decode) and scan_fits(E, P) else {}
        ext.group(pick, ids, s.count, members, R, slots, E, pdl, **listed, **({"tile": tile} if tile else {}),
                  **scan)
    (nt, w, sk, pf, rt), dec = gu
    staged = {"staged": True} if sw.expert_staged and many and stageable(gu, ex.k2_gu, ex) else {}
    rot = {"rot_x": x, "rot_suh0": ex.suh_g, "rot_suh1": ex.suh_u, "rot_arrivals": s.rot_arrivals} \
        if staged and listed and not dec and sw.rot_fuse and rot_fits(x) else {}
    if not rot:
        ext.rot_in(x, x.stride(0), pick, ex.suh_g, ex.suh_u, s.xg, s.xu, R, D, slots, E, pdl)
    fuse_gu = sw.gateup_fuse and bool(listed) and fusable(gu, ex.k2_gu)
    fused = {"epi": EPI_GATEUP, "s0": ex.svh_g, "s1": ex.svh_u, "s2": ex.suh_d, "out": s.xd, "arrivals": s.arrivals,
             "limit": float(limit), "act_mode": act_mode} if fuse_gu else {}
    ext.grouped(s.xg, s.xu, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, s.z, 2, D, I,
                P, sk, slots, ex.cb, nt, w, pf, ex.k2_gu[0], ex.k2_gu[1], rt, ex.window, dec, pdl,
                **({} if dec else listed), **fused, **staged, **rot)
    if not fuse_gu:
        # a staged launch writes one total a matrix: its splits summed as the epilogue sums them
        ext.gateup_epilogue(s.z, pick, ex.svh_g, ex.svh_u, ex.suh_d, s.xd, R, P, I, 1 if staged else sk, slots, E,
                            float(limit), act_mode, pdl)
    (nt, w, sk, pf, rt), dec = dn
    into = s.y if y is None else y
    fuse_dn = sw.down_fuse and bool(listed) and wts is None and sk == 1 and fusable(dn, ex.k2_d)
    fused = {"epi": EPI_DOWN, "s0": ex.svh_d, "out": into} if fuse_dn else {}
    staged = {"staged": True} if sw.expert_staged and many and stageable(dn, ex.k2_d, ex) else {}
    ext.grouped(s.xd, s.xd, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, s.z, 1, I,
                D, P, sk, slots, ex.cb, nt, w, pf, ex.k2_d[0], ex.k2_d[1], rt, ex.window, dec, pdl,
                **({} if dec else listed), **fused, **staged)
    zsk = 1 if staged else sk
    if wts is None:
        if not fuse_dn:
            ext.down_epilogue(s.z, pick, ex.svh_d, into, R, P, D, zsk, slots, E, pdl)
        return s.y[:P] if y is None else y
    if out is None:
        out = torch.empty((R, D), dtype=torch.float32, device=x.device)
    # the down epilogue and the combine in one launch (the same arithmetic in the same order as the two)
    ext.down_combine(s.z, pick, ex.svh_d, s.y, wts, out, R, P, D, zsk, slots, E, pdl)
    return out


Launch = tuple[tuple[int, int, int, int, int], bool]


def scan_fits(experts: int, pairs: int) -> bool:
    """Whether group_scan_kernel takes ``experts`` experts and ``pairs`` picks (else group_kernel groups them)."""

    return experts <= SCAN_EXPERTS and pairs <= SCAN_PICKS


def fusable(launch: Launch, k2: tuple[int, int]) -> bool:
    """Whether a fused epilogue has an instance for ``launch`` at the rate codes ``k2`` (min, max): grouped_kernel
    (not decode_kernel) at the prompt setting, rates up to 5 bits or 2.25 bits alone. The 2..16 instances spilled
    registers (ptxas, sm_121), so a layer with experts above 5 bits keeps the epilogue kernels."""

    (nt, w, _, pf, rt), dec = launch
    lo, hi = k2
    rates = (lo, hi) in ((8, 8), (RATE_WORDS + 18, RATE_WORDS + 18)) or (lo >= 2 and hi <= 10)
    return not dec and (nt, w, pf, rt) == (8, 4, 1, 2) and rates


def stageable(launch: Launch, k2: tuple[int, int], ex: Exl3RoutedExperts) -> bool:
    """Whether TF_EXL3_EXPERT_STAGED has an instance for ``launch``: the fused epilogues' setting and rates (their
    staged k steps fit static shared memory), on trellises that start on 16 bytes."""

    return ex.aligned and fusable(launch, k2)


def rot_fits(x: torch.Tensor) -> bool:
    """Whether TF_EXL3_ROT_FUSE's 16-byte loads take the rows of ``x``: bf16 or fp16, each row on 16 bytes."""

    return (x.dtype in (torch.bfloat16, torch.float16) and x.dim() == 2 and x.stride(1) == 1
            and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0)


def listed_tile(gu: Launch, dn: Launch) -> int:
    """The member group's rows (16 a row tile) the item list takes for ``plan``'s two launches: 0 when both run on
    decode_kernel, which takes no list, or when their grouped_kernel launches group members differently."""

    tiles = {16 * cfg[4] for cfg, dec in (gu, dn) if not dec}
    return tiles.pop() if len(tiles) == 1 else 0


def plan(cfg_gu, cfg_d, I: int, D: int, R: int, many: bool, decode: bool) -> tuple[Launch, Launch]:
    """((nt, warps, splits, pf, row tiles), decode_kernel?) for gate/up (K D -> N I) and down (K I -> N D).

    A prompt window takes prompt_config; a window of at most DECODE_ROWS rows with ``decode`` (TF_EXL3_EXPERT_DECODE)
    takes decode_kernel at decode_config's tile; the rest keep grouped_kernel at their cfg."""

    def one(cfg, K, N):
        if many:
            return prompt_config(cfg), False
        dc = decode_config(cfg, K, N, R) if decode else None
        return ((*dc, 1), True) if dc else ((*cfg, 1), False)

    return one(cfg_gu, D, I), one(cfg_d, I, D)


def dequant(trellis: torch.Tensor, codebook: int | str, window: int = 16) -> torch.Tensor:
    """W_q [K, N] fp16 of one matrix through the kernels' own lane decode (ExLlamaV3's ``reconstruct``); for tests."""

    cb = codebook_id(codebook) if isinstance(codebook, str) else int(codebook)
    k2 = k2_of(trellis)
    K, N = trellis.shape[0] * 16, trellis.shape[1] * 16
    out = torch.empty((K, N), dtype=torch.float16, device=trellis.device)
    _ext().dequant(trellis.contiguous(), out, k2, cb, check_window(window))
    return out
