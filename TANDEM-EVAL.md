# TandemLLM ideas in the TensorFold fork: evaluation

Six of the seven tasks (T1, T2, T3, T5, T6, T7) are implemented on branch `feat/tandem-ideas` (off `fork/dev`,
`9dbd143b`), uncommitted. Each sits behind an environment flag that is off by default, so the served path runs the
same code as before unless a flag is set. T4 (a merged copy and MTP tree) is not implemented: it needs tree verify in
five places that assume a chain, listed under T4. No speed number here was measured on the GPU. The live server held
the GPU for the whole session (2026-10-14), so every speed figure below is simulated and says so. The exactness claims
rest on tiny-model GPU tests, and `tools/tandem_ab.py` is ready to measure speed when the box is free.

Spec: the "TensorFold tasks" table in `~/workspace/tandemllm-ideas.md`. Reference code: `~/workspace/TandemLLM`
(AGPL-3.0, commit `e2a393a9`), read for algorithms and thresholds, never run, no code copied.

| task | flag (off by default) | state |
|-|-|-|
| T1 StairCut cut | `TENSORFOLD_STAIRCUT=1` | done; simulated +0.8% to +5.5% on mixed text |
| T2 never slower than plain | `TENSORFOLD_PLAIN_GATE=1` | done; simulated gain only on sustained hard text at long context |
| T3 idle graph capture | `TENSORFOLD_IDLE_GRAPHS=1` | done for `--parallel 1`; production (`--parallel 4`) runs no decode graphs |
| T4 copy and MTP tree | none | not done; plan below |
| T5 end and grid anchors | `TENSORFOLD_END_ANCHOR=1`, `TENSORFOLD_GRID_ANCHORS=4096` | done; TTFT gain unmeasured |
| T6 token loop detector | `TENSORFOLD_THINK_LOOP=1` | done; 7 of 183 saved reasoning blocks fire |
| T7 self-check and per-drafter metrics | `TENSORFOLD_SELF_CHECK=1`, `TENSORFOLD_DRAFTER_METRICS=1` | done; self-check passes on the tiny model |

## How the tests were run

The host `.venv` has no torch, so every test ran in the serving image `tensorfold-run:ffdaafceb2f0` with the worktree
mounted at its host path, in containers named `tftest-*`, with `PYTHONDONTWRITEBYTECODE=1`:

```
docker run --rm --name tftest-x -e PYTHONDONTWRITEBYTECODE=1 \
  -v /home/agis/workspace/TensorFold-tandem:/home/agis/workspace/TensorFold-tandem \
  -w /home/agis/workspace/TensorFold-tandem tensorfold-run:ffdaafceb2f0 bash -c \
  'pip install --target /tmp/pt pytest pytest-timeout >/dev/null && \
   PYTHONPATH=/tmp/pt:src:tests python -m pytest -p no:cacheprovider --timeout 300 -q <tests>'
```

GPU tests added `--gpus all --ipc=host` and ran only the tiny-model files under `tests/cuda/` that the changes touch,
each time after `free -g` showed 37-38 GiB available. The 33-minute GPU suite was not run.

Baseline, CPU-only, on `fork/dev` before any change (2026-10-14): 4,261 passed, 129 failed, 11 errors, 187 skipped
(`tests/streaming` excluded, as the memory notes advise). The suite delta after the changes is under "Test suite".

## T1: StairCut cut for the MTP chain

**What changed.** `families/qwen4_exp/cuda/staircut.py` (new) holds the rule. A round that verifies the first `n`
drafts commits `1 + q_1 + ... + q_n` tokens on average, where `q_k` is the product of the first `k` drafts' calibrated
probabilities. The cut keeps the `n` that maximises that over the round's price: the verify of `n + 1` rows plus one
MTP step per draft. Calibration maps the head's probability to the landing rate per decile, `(hits + 4p) / (n + 4)`,
halving a bin's evidence past 400. Prices are a table per context class (0 up to 8,192 tokens, then the next power of
two, the graphs' buckets), EMA 0.1 online from where the prior puts them, never falling with rows, and a round over 4x
its price teaches nothing. The prior is timed at warm-up: each window of 1 to depth + 1 rows, the median of 5 runs.

`decode.draft_cut` replaces `decode.draft` when the flag is on, in all three decoders: `mtp_decode` (`--parallel 1`),
`MultiDecoder` shared rounds (`--parallel N`, the production path) and the lone graph slot. Prices learn only from
rounds one stream ran alone; calibration learns from every verified window. Two ranks ignore the flag, because their
windows must agree and the prices follow each process's clock.

One departure from the spec. The spec's chain is drafted to full depth and then cut. Here the chain stops drafting
when deeper drafts are not expected to raise the score, each landing as often as the last one's calibrated chance. An
optimistic bound (each deeper draft lands as often as the deepest so far) drafted levels the cut then dropped. At
3 ms an MTP step it lost 2.6% to the fixed cut in simulation. The expected-value rule gained 2.0% there.

**Tests.** `tests/test_flashnext_staircut.py` (CPU, 14): prices, calibration, the cut against a brute-force argmax
over 256 probability vectors, the stopping rule, the off-by-default flags. `tests/cuda/test_flashnext_staircut_decode.py`
(GPU, tiny model, 12): with the cut on, `mtp_decode` (eager and graphs, depth 2 and 6, greedy and sampled), shared
rounds and the lone slot all give the serial reference's tokens, windows reach the full depth, and warm-up pricing
leaves decoding exact.

**Simulated evaluation** (`python3 tools/staircut_sim.py --check --tokens 40000`, 2026-10-14). Every policy decodes
the same synthetic trace, so they differ only in the rows they verify. Verify prices come from TandemLLM's
`flashnext-3.05.dgx-spark.json` (dated 2026-10-12, marked provisional there). The trace's head was fitted by grid
search to the live server's counters on 2026-10-14 (198,270 of 278,764 drafts accepted, 71.1%, about 3.06 tokens a
round under the 0.7 cut). The fixed rule on the fitted trace gives 70.5% accepted and 3.07 tokens a round. The MTP
step's ms on Flash Next is not measured, so the step price is swept.

| mixed text, tok/s (simulated) | class 0 | 32k | 128k | 256k |
|-|-|-|-|-|
| fixed 0.7 cut, step 1.5 ms | 79.1 | 67.5 | 53.6 | 50.4 |
| StairCut, step 1.5 ms | 81.6 (+3.2%) | 69.2 (+2.5%) | 55.3 (+3.2%) | 52.8 (+4.7%) |
| StairCut, step 0.5 ms | +3.9% | +3.0% | +4.4% | +5.5% |
| StairCut, step 3.0 ms | +2.0% | +1.5% | +2.4% | +3.1% |
| StairCut, a calibrated head (p ** 1.0) | +1.2% | +0.8% | +1.1% | +2.1% |
| StairCut, seed 2 | +3.2% | +2.6% | +3.7% | +5.3% |

By text kind at step 1.5 ms, the cut gains least on json (+0.0% to +0.5%), where the fixed rule already verifies
deep chains, and most on hard text (+3.9% to +7.7%).

**Unverified.** All of it on the real model: the gain, the warm-up prior's accuracy, and the host time the cut adds per
round (a Python loop over at most 7 drafts). TandemLLM reports 49.89 against 47.13 tok/s (+5.9%) on its 27B and has
no Flash-Next A/B against a fixed cut.

## T2: never slower than plain

**What changed.** `PlainGate` in `staircut.py`. Per context class it keeps a drafted round's cost as a ratio of two
EMAs (ms and committed tokens, weight 0.05) and a plain round's ms (EMA 0.1). While drafting costs more a token than
the plain step, the class takes plain rounds: the anchor alone, with the MTP head still absorbing the kept rows, so the
head never falls behind. It drafts one probe round every 16 rounds, an interval that doubles to 256 while drafting
keeps losing.

Three additions came out of the simulator:

- A class's first 16 drafted rounds are summed before the EMA starts, and the gate waits for them. One slow first
  round otherwise sent a class plain, and 5%-weight probes took hundreds of rounds to bring it back.
- The class goes plain only when drafting costs 10% more than the plain step. The EMA of rounds that commit 1 to 7
  tokens wanders, and a class that drafted 13% cheaper than plain went plain at the first excursion.
- The gate measures the plain step itself, with one plain round after a class's first 16 drafted rounds and again
  every 256. The fixed cut never verifies a one-row window, so without this the plain cost was a prior that the long-
  context price shift distorted, and the gate never acted.

**Tests.** CPU (in `test_flashnext_staircut.py`): the probe pattern (plain, then a probe after 16 rounds, then after 32),
the ratio of EMAs, the first-16 rule, the plain-step measurement. GPU: a gate that has learned drafting loses gives
the serial tokens with 30 or more of 48 rounds plain, and the head's cache follows every committed row
(`mtp_len == pos`). Shared rounds with the gate on equal each stream alone.

**Simulated evaluation**, same runs as T1, plain-step ceiling 38.2 to 42.6 tok/s by class:

| tok/s change vs the fixed cut (simulated) | 0 | 32k | 128k | 256k |
|-|-|-|-|-|
| gate alone, hard text | -0.1% | -0.3% | +7.7% | +15.7% |
| gate alone, mixed text | -0.0% | -0.0% | -1.6% | -3.5% |
| StairCut and gate, mixed text | +2.9% | +2.3% | +1.5% | +1.6% |
| StairCut alone, mixed text | +3.2% | +2.5% | +3.2% | +4.7% |

The gate pays only where drafting loses for long stretches: hard text at 128k and above, where the fixed cut runs at
32.9 to 35.6 tok/s against a 38 tok/s plain step. On mixed text it costs 1.6% to 3.5% at long context (up to 8.0%
with a calibrated head), because the text changes faster than its probes. Leave it off unless a workload is
long-context and hard to draft for.

**Unverified.** Whether drafting ever loses on our engine: in the simulation it loses only because TandemLLM's price
file makes a 2-row verify at 256k cost twice the plain step, which our kernels may not share.

## T3: capture missing CUDA graphs while idle

**What changed.** `cuda/idle_work.py` (new) runs GPU work between requests: one second after the last request, a
worker thread plans the steps and runs them one at a time under the lock a request holds, and stops before a step
when a request waits or one ran since the plan. `Graphs.missing` and `Graphs.idle_steps` list and capture the decode
graphs (every window size warm-up captured, both DeltaNet parities, and the MTP head's) of the context bucket the last
reply ended in and the next one up. A capture writes only rows past the committed ones and the live recurrent
buffers. A next request restores those from a kept snapshot (a clone) or resets them, which `State.snapshot` and
`prefill_begin` confirm. A failed capture (`RuntimeError`) turns idle work off, and requests then capture lazily as
before.

**Scope.** Decode graphs exist only with `--parallel 1` (and the lone slot under `TENSORFOLD_LONE_GRAPHS=1`). The live
server runs `--parallel 4` with eager rounds, so T3 changes nothing there. Two ranks ignore the flag: a capture holds
NCCL calls, which one rank cannot record alone.

**Tests.** `tests/test_idle_capture.py` (CPU, 3): the delay, a request that waits stops the steps after the one
running, and a new plan picks up after it. `tests/cuda/test_flashnext_idle_graphs.py` (GPU, 2): after idle captures
at a 40-token context, an 8,300-token request in the 16,384 bucket captures nothing mid-reply and gives the serial
reference's tokens, greedy and sampled.

**Evaluation.** Not measured. The engine's own docstring records about 0.155 s a window size to record graphs (EXL3
3.05bpw, `--parallel 4`, 2026-10-11), so a reply crossing into a new bucket at depth 6 stalls for about 21 captures
(7 window sizes at 2 parities, 7 MTP sizes), about 3.3 s. TandemLLM measured 3.4 to 5.1 s for the same stall.

## T4: copy and MTP drafts in one tree (not implemented)

The DeltaNet side is ready: `cuda/kernels/gdn.py` `tree()` runs the recurrence over any parent list, and
`gdn_multi.Tables` already folds non-contiguous kept rows into the next round. Five other parts assume a chain:

1. Attention (`attention.py`, `attn_multi.py`, Triton): rows attend by position, so a branch row would see its
   siblings. Tree verify needs an ancestor mask in the dense kernel and in the sparse index and pooled path.
2. RoPE: a tree row's position is its depth, while its cache slot is its row, and the kernels tie the two together.
3. Conv taps (`gdn_multi.Tables.win`): each row reads the three rows before it in the window; a tree row needs its
   three ancestors.
4. The n-gram embeddings (`ple_tail`, `ple_nrow`, `shift_windows`): a row's n-gram is the tokens before it; a tree
   row needs its ancestors' tokens.
5. Commit (`forward.commit`): keeps the first `keep` rows. A tree path needs its KV rows moved to contiguous slots, the
   indexer keys and pooled blocks rebuilt, and the conv and n-gram windows shifted along the path.

The drafting side (merging the copy chain into the MTP chain with `q = 1 - (1 - a)(1 - b)` on shared nodes) is a
small change to `staircut.py` once verify takes trees. Each of the five parts needs a bit-equality test against the
serial step on the tiny model, the same way `test_flashnext_forward.py` tests chains. TandemLLM reports edit 106 to
132 tok/s and copy 102 to 133 tok/s for it; our 2026-10-14 A/B found copy drafts slower than DFlash2 on GLM, so the
merged tree's value on Flash Next is open.

## T5: end and grid anchors

**What changed.** End anchor (`TENSORFOLD_END_ANCHOR=1`): when a reply finishes cleanly, the state where its
committed tokens end is kept like a prompt state, with the MTP head one row behind and the last row's stream as the
tail. A next turn that sends the reply back resumes after it. `mtp_decode(keep_end=True)` returns that state, and the
engine keeps it next to the prompt-end state. In shared rounds `MultiDecoder._keep_end` first folds the round's
deferred recurrent rows into the slot, then absorbs the last window's rows into the head, then keeps the slot. The
lone slot does the same.

The end state carries decode bits, and a prefill of the same tokens does not reproduce them. The fork's rule so far
was the opposite ("never decoded rows: prefill and decode bits differ", `cuda/streams.py`). With the flag on, a
resumed next turn is the model's continuation of the same text, and it is not bit for bit a cold run's. TandemLLM
accepts the same trade.

Grid anchors (`TENSORFOLD_GRID_ANCHORS=4096`): prompt states are also kept at every multiple of 4,096 tokens, and a
line within 256 tokens of a message marker gives way to the marker. They share the kept-state cap (4 entries with
`--parallel 1`, 8 with `--parallel N`), so on a long prompt the highest lines push out older entries.

**Tests.** `tests/test_grid_anchors.py` (CPU, 4). `tests/cuda/test_flashnext_end_anchor.py` (GPU, 4): on one engine
and in shared rounds (eager and the lone slot), a next turn that extends the reply resumes at the reply's committed
end (`cached == len(prompt) + len(committed)`), the head's row count after the resume equals a cold prefill's, and the
first token equals a cold run's.

**Evaluation.** Not measured. TandemLLM reports next-turn TTFT 112 to 131 ms against 1.6 to 1.8 s. Whether our clients'
resent turns match the anchor is unverified: it needs the chat template to render the previous reply exactly as the
model wrote it, reasoning included, which Qwen's template does only for the turns after the last user message.

## T6: token loop detector

**What changed.** `engine/think_loop.py` (new): three arms watch a think block's tokens with the spec's thresholds.

| arm | fires when |
|-|-|
| pattern | the last `max(4, ceil(128 / s))` blocks of `s` tokens are equal, `s` from 1 to 16 |
| history | past 2,048 tokens, every 64th, at least 70% of the last 256 tokens' 16-grams occurred before them, on two checks in a row |
| novelty | past 320 tokens, every 64th, under 10% of the last 128 tokens' 8-grams are missing from the 128 before them |

`LoopGate` follows the call gate's interface (`cut`, `observe`), so it works on the run-again path and on the inline
splice path that `--parallel` uses. The firing token becomes Qwen's budget-forcing phrase, `</think>` and a blank line,
once a reply. The model reads the phrase; `think_loop.hide` takes it out of the client's reasoning text, holding back
a tail that may be its start. The reply's `tensorfold` block records the arm and the token (`think_loop`).

**Tests.** `tests/test_think_loop.py` (CPU, 18): each arm fires at the exact token its rule names (pattern sizes 1, 2,
3, 7 and 16; the history arm's two-in-a-row; a 100-token cycle for novelty), varied text never fires, a lookahead
changes nothing, the server closes a looping block with the phrase hidden (stream and non-stream, widths 1 and 4), the
inline path writes the close once, and the flag off leaves a looping reply alone.

**Offline evaluation** (`tools/think_loop_scan.py`, 2026-10-14) over saved eval transcripts in `~/work/tfeval`,
`eval`, `e46eval`, `exl3eval` and `ext-runs`, tokenized with the served tokenizer: 183 distinct reasoning blocks of
2,000 characters or more (599,019 tokens).

| | blocks | fired |
|-|-|-|
| under 2,048 tokens | 108 | 0 |
| 2,048 to 8,191 | 74 | 7 |
| 8,192 or more | 1 | 0 |

Four fires (pattern arm) are genuine degenerate runs, all in CODE-01 (`float regex accepts '0.000...'` over 128 or
more zeros). Three (history arm) fired on code the reasoning wrote a second time, in RR-02, CODE-13 and CODE-06, which a
person may or may not call a loop. All 7 replies scored 1.0 in their runs, and 5 of the 7 blocks ran to the 8,000-token
thinking budget anyway. The corpus has no labelled loop set, so the catch rate is unmeasured. TandemLLM reports 3 of 3
failed turns caught and 4 of 80 healthy long turns closed.

## T7: self-check at load, and per-drafter metrics

**What changed.** Self-check (`TENSORFOLD_SELF_CHECK=1`, `families/qwen4_exp/cuda/self_check.py`): after warm-up the
engine decodes 32 greedy tokens one row at a time, recording each logits row, then replays the same tokens through
windows of every verify width from 2 to depth + 1 and compares every row with `torch.equal`. It runs on the path the
server uses (the one-stream engine with its graphs, or the shared rounds' `compute` with deferred recurrent commits).
On any difference the log names the width and row, and every request is served without drafts (the serial
reference); `splices` turns off with it. Two ranks skip the check.

Per-drafter metrics (`TENSORFOLD_DRAFTER_METRICS=1`): streams count verified and accepted drafts by the drafter that
wrote them (`mtp`, `copy`, `fix` for a gate's forced tokens). `/metrics` then adds
`tensorfold:draft_tokens_total{drafter=...}` and `tensorfold:draft_accepted_tokens_total{drafter=...}`. The families
appear only once an engine reports them. The existing `mtp_drafted_total` and `mtp_accepted_total` are unchanged.

**Tests.** `tests/test_flashnext_self_check.py` (CPU, 3): equal bits pass; one ulp in one row of width-3 windows fails
and names width 3, row 1. `tests/test_drafter_metrics.py` (CPU, 3). `tests/cuda/test_flashnext_tandem_checks.py`
(GPU, 5): the self-check passes on the tiny model's real kernels for the one-stream engine (eager and graphs) and for
shared rounds. With an oracle copy index (the tiny model's reply never repeats, so a real index finds nothing),
per-drafter counts sum to the stream's totals and every copy draft lands.

**Evaluation.** Not measured on the real model. The check costs about 32 one-row steps plus 6 widths' windows over 32
tokens, about 90 forwards, which at about 25 ms a forward (TandemLLM's 1-row price) is about 2 to 3 s at load
(estimate).

## Test suite

Full CPU-only suite after the changes (2026-10-15, after the review fixes): 4,324 passed, 129 failed, 11 errors, 187 skipped. The 129 failures
are the baseline's 129, test for test, and the 11 errors (`tests/test_affine_rows.py`) also occur on a clean `fork/dev`
copy. The 63 more passes are the new CPU tests.

New test files: 11 (7 CPU, 4 GPU); new tests: CPU 63, GPU 23. The review fixes' `multi.py` change (the plain gate measures only on a round one stream drafts alone) has not run on the GPU. GPU files run, all on the tiny model (2026-10-14):
`test_flashnext_staircut_decode.py` (12 passed), `test_flashnext_tandem_checks.py` (5), `test_flashnext_idle_graphs.py`
(2), `test_flashnext_end_anchor.py` (4), and the existing `test_flashnext_multi.py` and `test_flashnext_forward.py`
(93 passed together with the StairCut file before the T3 to T7 changes). The final run on 2026-10-15 took 12 tiny-model
files (the 4 new ones and `multi`, `forward`, `kvquant`, `prompt_cache`, `fill_arrivals`, `fork_lanes`,
`graph_capacity`, `large_pieces`): 217 passed, 1 skipped, in 6 minutes. The two-rank files (`test_flashnext_tp*.py`)
were not run; every new behaviour is off on two ranks.

Lint (`ruff check --select E,F,W`, ruff 0.17.0) on the changed files: the same findings as `fork/dev` (long lines and
three unused imports that predate this branch), none new.

## GPU A/B commands, for when the box is free

Stop the serving engines first: `tools/tandem_ab.py` boots a full model, refuses to start below 90 GiB of
MemAvailable, and refuses ports 30000, 30001 and 4000. It runs A, B, A, greedy and T=0.7 (seed 1234), on prose, code,
json, edit and copy prompts, prints median decode tok/s by boot and the B/A geometric mean, and lists every reply whose
text differs from boot 0's. Keyed sampling makes the text identical at both temperatures for a flag that only moves
speed.

```
cd ~/workspace/TensorFold-tandem
M=/home/agis/models/Qwen3.8-Flash-Next-exl3-3.05bpw
# T1: StairCut against the 0.7 cut, the production shape (--parallel 4)
python3 tools/tandem_ab.py --model $M --arms 'A=;B=TENSORFOLD_STAIRCUT=1' --order A,B,A --reps 3 --out ab-t1.json
# T1 at --parallel 1 (graphs on)
python3 tools/tandem_ab.py --model $M --parallel 1 --arms 'A=;B=TENSORFOLD_STAIRCUT=1' --reps 3 --out ab-t1-p1.json
# T2: the gate on top of StairCut at long context
python3 tools/tandem_ab.py --model $M --context 262144 --long-tokens 120000 --kinds prose,code \
    --arms 'A=TENSORFOLD_STAIRCUT=1;B=TENSORFOLD_STAIRCUT=1,TENSORFOLD_PLAIN_GATE=1' --reps 2 --out ab-t2.json
# T3: idle graph captures, a 12k prompt crossing into the 16,384 bucket (--parallel 1)
python3 tools/tandem_ab.py --model $M --parallel 1 --long-tokens 12000 --kinds prose \
    --arms 'A=;B=TENSORFOLD_IDLE_GRAPHS=1' --reps 3 --out ab-t3.json
# T7: self-check and per-drafter counters (the self-check line is in ab-t7.json.B1.log; the counters in the JSON)
python3 tools/tandem_ab.py --model $M --arms 'A=;B=TENSORFOLD_SELF_CHECK=1,TENSORFOLD_DRAFTER_METRICS=1' \
    --reps 1 --out ab-t7.json
```

T5 and T6 need multi-turn and thinking-on traffic, which the A/B script does not send. For T6, run a TrueScore or
spark-bench pass with `TENSORFOLD_THINK_LOOP=1` and grep the server log for `think loop closed`. For T5, send a tool-call
turn and its follow-up with `return_token_ids` and read `cached` in the follow-up's `tensorfold` block.
