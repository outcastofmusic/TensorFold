# Nemotron 3.5 Super VL

`run.sh` serves NVIDIA's Nemotron 3.5 Super VL 120B-A12B (`nemotron_h_omni`) with upstream TensorFold's native 1.1.0
server, which `run.sh` runs in place of this checkout's Python engine. It is an opt-in swap target. The default stays
`qwen-flash-next`. Upstream's [recipe](https://github.com/ashhart/TensorFold/blob/v1.1.0/docs/recipes/nemotron-3.5-super-vl.md)
describes the engine, its exactness and its speed.

## Run

```bash
./prepare.sh nemotron-3.5-super   # nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 at 82d93699 (75.1 GB), and the
                                  # 1.1.0 linux-aarch64 archive, sha256-pinned, into ~/models/tensorfold-1.1.0-linux-aarch64
./run.sh nemotron-3.5-super       # served as nemotron-3.5-super on 127.0.0.1:30000, or through ./run.sh swap
```

`run.sh` mounts `NATIVE`'s `bin/` and `lib/` into the base image and starts:

```bash
tensorfold-native serve ~/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 --name nemotron-3.5-super \
  --context 131072 --max-tokens 131072 --precision checkpoint --vision --thinking-budget 8000 --no-update-check
```

MTP drafts are on, the server's default. `CONTEXT`, `THINK_BUDGET` (0: none) and `EXTRA` change the flags as for the
other targets. Every flag above exists in both 1.0.12 and 1.1.0 (`zig/src/server/cli.zig`).

1.1.0 makes `--mtp-drafts N` and `--mtp-confidence C` reach CUDA Nemotron streams (#997). `N` caps the MTP depth at
up to 31 for Super, and 0 turns drafting off. `C`, from 0 to 1, replaces the cost-based cutoff with a cumulative
confidence threshold. `run.sh` passes neither, so the server's cost-based depth applies. `EXTRA` can pass them for an
A/B.

1.1.0 also answers `response_format` (JSON schema) and `tool_choice: "required"` through `lib/libtfgrammar.so`, which
1.0.12 refused (#951). The release archive has neither `libturbojpeg.so.0` nor `libwebpdecoder`. The server therefore
refuses JPEG and WebP images and serves PNG. A `lib/` holding libjpeg-turbo 3.1.4.1's `libturbojpeg.so.0`, such as a
local build's `zig-out/native`, lets JPEG serve through `NATIVE=`. WebP needs libwebp 1.6.0's `libwebpdecoder` there
too.

## Numbers

| Measure | Value | When |
| --- | --- | --- |
| TrueScore 76x2, thinking off | 94.1 weighted, 91.7 unweighted (Calibration 100) | 2026-10-18, `tf-runs/bench2` |
| Planning x4, thinking off | 62.6 | 2026-10-18, `tf-runs/bench2` |
| Ready after start | 44 s | 2026-10-18 |
| Available after load at 131,072 tokens | 40 to 41 GiB of 121.7 | 2026-10-18 |

The binary in these runs was a local build of upstream `6ee828c2`, which changes no engine code from the 1.0.12
release. `tf-runs/gpu-super110/run.sh` repeats the speed, memory and greedy-reply checks on 1.1.0 against that build.

## Memory

The weights take 74.2 GiB with the image tower, as the server logged at load on 2026-10-18
(`tf-runs/bench2/super.server.log`, `--vision` on). One stream's KV and Mamba state take 1.28 GiB at 131,072 tokens.
The 40 GiB left after load holds d1-3B (`D1=1`), whose estimate is 7.3 GiB. Clef stays off, because it guards Qwen's
thinking.

## Precision

`--precision checkpoint`, the Blackwell default, runs the checkpoint's own math: e4m3 rows into the FP8 layers and
NVFP4 rows into the experts. Its replies differ from `--precision full`'s bf16 rows. Upstream measured a mean
KL(full || checkpoint) of 0.0077 nats over 768 teacher-forced tokens, with the same top token on 98.4% of them.
`checkpoint` reads a cold 8,192-token prompt 1.81x faster.

## Swap and gateway

`/v1/models` reports `owned_by: "tensorfold"`, so the gateway's `reasoning_loop` act applies. The native server has no
`/internal/persist` route and answers 404, which `swap.py` treats as a model without disk states. A swap away from
Super therefore stops it at once, and its kept prompt states stay in memory only.

## Served check

`tf-runs/super/smoke.py` checks plain chat, thinking split into `reasoning_content` and `content`, a tool call, a
tool result answered, and a streamed call whose arguments parse as JSON.
