# Gotchas that produce *plausible wrong numbers*

The dangerous bugs are not the ones that crash. These all ran to completion and
returned numbers a reasonable person would believe.

## Silently randomly-initialized weights

`transformers` 4.52 renamed Qwen2.5-VL's submodule path
(`model.layers.*` → `model.language_model.layers.*`). Any **subclass** that
bypasses the rename mapping loads **zero** pretrained weights, prints a warning,
and carries on.

**Narrowed by a second data point (thanks to the frame-selection line): the
hazard is SUBCLASSING, not the version.** Using
`Qwen2_5_VLForConditionalGeneration` directly on transformers 4.57.6 is fine —
they reproduce published LongVideoBench numbers (59.84 @64 frames, n=1337), which is
impossible with random weights. So do **not** pin 4.51.3 reflexively; pin only if
you are loading through a subclass that reimplements `forward` (CLaMR's
`ColQwen2_5` does). The general check below is what matters, not the version.

Result was nDCG@10 0.2244 instead of 0.5847 — low, but not obviously broken,
because identical tokens still get identical *random* embeddings, so
late-interaction degenerates into random-projection lexical matching.

**Check:** grep model-loading output for `newly initialized`, and compare
checkpoint keys to `model.state_dict()` keys before believing any number.

## Left padding + last-token pooling

VLM2Vec sets `padding_side="left"` and pools "the largest index where
`mask == 1`". Using `mask.sum(1) - 1` (the token *count*) is correct **only for
the longest sequence in the batch**; everything else pools a pad position.

**Symptom:** cosine to reference embeddings is ~0.99 for one row in the batch and
0.3–0.5 for the rest.

## Visual inputs dropped by a tensor filter

Some processors return `pixel_values` as a **list of numpy arrays**, not a
tensor. A filter like `{k: v for k, v in inputs.items() if torch.is_tensor(v)}`
silently deletes the images, and every candidate encodes to the same text-only
vector.

**Symptom:** pairwise cosine among embeddings of *different* images ≈ 0.999
(reference: ~0.35). Always sanity-check that different inputs give different
embeddings.

## Any max-over-alternatives quantity needs a null at the same cardinality

Found independently by two lines, in two different shapes. Report the quantity
**minus its null**, never the raw quantity.

**Oracles.** On multiple choice the null harvests the guessing rate: a subset
oracle over 6 *random* draws reached 73.49 where 6 *real* selection methods
reached 73.76 — a claimed ~16 points of headroom was almost entirely lottery.

**Flip sets.** Same trap, different shape, and easy to miss because no oracle is
involved. "Items that budget B gets wrong and budget 64 gets right" looked like
10 points of headroom on LongVideoBench — but two equal-strength 16-frame runs
already churn **8.12** points by guessing alone, leaving ~2.5. Measure the churn
floor with independent draws at the same budget.

**Retrieval.** The guessing floor is usually too small to harvest (top-10 of 1504
= 0.66%), but the null still earns its keep by exposing error **correlation**:
four real MultiVENT channels scored **12.7 points below** four noise-matched
noisy copies of a single channel. Redundancy isn't "they overlap" — it's "they
share a failure mode".

Check it against the floor's **distribution**, not its mean. On LongVideoBench
the 56 ordered pairs span 6.76–9.84 (sd 0.63), which puts the observed
`recover` of 10.66 above every pair including the 9.84 maximum, and the observed
`lost` of 7.89 below 42 of the 56 — inside the floor, i.e. more frames churn but
do no systematic harm. Quote **exceedance counts, not z-scores**: 8 runs put
each run in 14 pairs, so the pairs are dependent and a z overstates.

⚠️ That `lost` term is the **only** internal check a flip-set correction gives
you. It is tempting to add "and the corrected ceiling closes on the
large-budget accuracy", but that is the same check twice: since
`acc@64 = acc@16 + recover − lost`, the quantity
`corrected_ceiling − acc@64` equals `lost − floor` identically. Both print
−0.23 by construction, not by agreement, and reporting the pair as independent
confirmation overstates the evidence. (We did exactly that for a few hours.)
For real confirmation, rebuild the floor at a second budget and require the two
corrections to agree.

## Precision is not significance

A grid too coarse to contain the optimum gave the wrong answer (w=0.95 -> 12.81).
Refining it gave a sharper wrong answer: w=0.93 -> 12.86, published as a
correction, when a paired bootstrap puts 0.93 vs 0.95 at +0.049 with CI
[-0.028, +0.123]. A denser grid buys resolution, not significance.

Before reporting an argmax as a finding, test it against its neighbours on the
same items. If the curve is flat, report the plateau. And note the argmax's
advantage over a baseline inherits the selection -- quote the plateau's WORST
endpoint, not the peak's: here the peak claimed +0.25 [+0.11, +0.40] while the
selection-free floor was +0.194 [+0.017, +0.373].

## A run can be correct while its sentence is wrong

A control measured exactly what it should have and was then described using a
retracted model, because the expected value had been withdrawn after the script
was written. The number was right, the framing around it was not, and the framing
is what a reader takes away.

When a run is anchored on a predicted value, re-check that the prediction is
still current immediately before reading the result -- the interval between
writing a control and running it is where landmarks go stale.

## Sweep the operator the benchmark actually uses

A fusion sweep ran for a full day on `w*(q.v) + (1-w)*(q.a)` while the incumbent
was `q . normalize(w*v + (1-w)*a)`. The two families agree at w=0 and w=1 -- so
both endpoints reproduced, every calibration check passed, and the whole middle
of the curve was a different operator. It cost a wrong bar (12.86 vs 13.12) and
an audio margin that was half its real size (+0.254 vs +0.474).

Endpoint agreement is not family agreement. Before sweeping a hyperparameter of
an operator, write the incumbent's formula out and check your parameterization
reduces to it at the incumbent's setting -- here, at w=0.5 the linear family
gives 5.00 where the incumbent gives 5.86, which was visible in the table the
whole time and read as an interesting side result instead of a red flag.

## Agreement between two lines is not replication

Two sessions independently found FLARE's best fusion weight at w=0.95 -> 12.81,
one by grid sweep and one by 5-fold holdout, matching to the decimal. It was
wrong: neither grid contained 0.93. Independent *implementations* are not
independent *designs* -- when both inherit the same grid, tolerance, or
preprocessing, agreement only confirms they made the same choice, and cannot
detect an error in that choice.

Before citing agreement as confirmation, name the thing that would have had to
differ for the two runs to diverge. If nothing does, it is one run reported
twice. For a swept scalar, refine the grid around the winner before publishing
the optimum -- and then check the peak is resolvable at all: here the paired
bootstrap on 0.93 vs 0.95 was +0.049 with CI [-0.028, +0.123], so the "corrected"
optimum was itself false precision and the honest answer is a plateau.

## sd exactly 0.00 means your seeds did nothing

A 5-seed robustness check returned 12.49 five times, sd 0.00 -- and nearly got
reported as stability. The fit was full-batch deterministic Adam; the seed
perturbed only a 1e-3 init the optimiser erased. One solution found five times
reads exactly like five agreeing samples.

If a seed sweep returns zero variance, the pipeline is deterministic and the
check measured nothing. Find the real source of variance -- for a deterministic
fit on fixed data that is the data, so use a paired bootstrap over items
(both arms score the same items, so compare per-item differences, not two
marginal CIs).

## Single-seed results

Seed sd was ~1 point on MMEB VisDial reranking. A first-seed "+2.3 points"
became **+0.27, p=0.22** at n=16. See [03](03-mmeb-visdial.md).

Prefer **paired** designs — same seed, same data order, one thing varied — which
make the difference far less noisy than two independent runs.

## Convex pooling cannot beat a max

`softmax(logits) · row ≤ max(row)`, always. If your learned reduction is a convex
combination and your baseline is `max`, the experiment is unwinnable by
construction and a flat sweep tells you nothing about the world. Use a zero-gated
**unbounded** residual: `score = max(...) + g · Δ`, `g = 0` at init.

## Zero-init deadlock

Zero-initializing both the gate and the residual's output layer gives both zero
gradient forever. Zero exactly one side.

A second form: a zero-init *projection* does not deadlock, it silently starves
everything upstream. Our aggregation head showed the projection taking 100% of
the gradient share and the attention machinery feeding it exactly 0% — it would
have trained the projection over a frozen random head and produced a plausible
mediocre number. Check gradient *shares*, not gradient existence.

## Contrastive loss saturating on cosines

Cosine similarities in [-1, 1] make cross-entropy saturate at ≈ ln(N) with no
gradient. Add a learnable logit scale (CLIP-style, init ~20).

## Storage

`hf_hub_download(cache_dir=...)` returns a symlink into a blob store; unlinking
the returned path frees nothing. Use `local_dir=<tempdir>` and `rmtree` it if you
are streaming hundreds of shards through a shared volume.

## qwen-vl-utils video min_pixels floor

Independently hit by two projects, so worth the exact numbers. In
qwen-vl-utils 0.0.14, `VIDEO_MIN_TOKEN_NUM = 128`, so the per-frame floor is
`128 × 28 × 28 = 100,352` px. If you pass only `max_pixels` (e.g. `224*224 =
50,176` for small frames, or `3136` for 4-token thumbnails), `max < min` and
`smart_resize` raises `AssertionError: The max_pixels of image must be greater
than or equal to min_pixels` on **every** item.

The error text does not say which minimum it means. Fix: set **both** in the same
element dict — `{"min_pixels": 784, "max_pixels": 3136}` yields a real ~3-token
frame. And note there are two call sites in CLaMR (`process_combined` and
`process_frames`); patching one is not enough.

## Trainer side effects

HF `Trainer` auto-enables `wandb` when it is installed, and can fail at the
*logging* callback after a complete, expensive eval pass. `--report_to none`.

## Masking with `-inf` turns a KL term into NaN once batches are ragged

`p · (log p − log q)` with padding masked as `-inf` evaluates
`exp(-inf) · (-inf − finite)` = `0 · -inf` = NaN, which then poisons every
downstream gate. Invisible with fixed-size inputs, because nothing is ever
padded; fires on almost every step once elements have different lengths. Mask
with a large **finite** value and zero the padded terms before summing. Add a
regression test on a deliberately ragged batch — a smoke test at fixed size
cannot see this class of bug.

## Feature caches keyed on the wrong id

Check `len(unique ids) == len(items)` before trusting one. A popular
video-instruction set keys its annotations on *video* id, not item id: 31,008
questions collapsed onto 7,774 feature files, later questions silently
overwriting earlier ones, and a reward cache keyed on that id mixed rewards
across different questions of the same video. Training then runs happily on
inputs conditioned by the wrong question.

## A selection experiment must change *which* items are fed, never *how*

Feeding chosen video frames as a pre-decoded image list double-resizes them
relative to the library's own file-reading path: 94% prediction agreement with
the baseline instead of ~99% — a few points of apparent "method effect" that is
really preprocessing. Patch the reader to honour explicit indices *inside* the
normal path, and verify agreement on a slice before running anything.

## A subset oracle on multiple choice harvests chance

`oracle@R = max over R candidate subsets of per-item correctness` gives every
wrong item R independent ~25% lottery tickets, and the inflation grows with R —
it is far larger than the 25% floor and subtracting an analytic chance term does
not remove it. The control is a **null oracle**: max over R *random* subsets at
the same R. Report `oracle − null_oracle`.

We claimed ~16 points of selection headroom this way. With the control, six
random 16-frame subsets scored 73.49 against six real methods' 73.76 — the
headroom was ~98% lottery, and the honest effect was the R=1 gap of ~2 points.
The failure is silent: the inflated number looks like strong motivation for the
whole research direction.

## Identical numbers across conditions that should differ is a bug, not a finding

A budget sweep at 2 and 4 frames returned running accuracy of 0.5741, 0.5711,
0.5705 — **the same digits at every checkpoint, in both jobs**. Cause: the eval
script accepts both `--budget` and `--frames-json`, and the frame file silently
wins. Both jobs evaluated the same precomputed 16-frame subsets. It also
filtered the item list to the file's coverage, quietly dropping 361 of 976 items
and changing the denominator, so even the surviving number answered a different
question than the one asked.

Nothing crashed and the logs looked healthy. The only signal was the digits
matching, which is why it is worth stating as a rule: **two conditions that
should differ and don't have a plumbing bug until proven otherwise.** Diff the
per-item predictions, not just the summary line.

The fix belongs in the script rather than in your memory — make the overriding
argument refuse the combination:

```python
sizes = {len(v["frame_indices"]) for v in frames_map.values()}
if sizes != {args.budget} and not args.allow_budget_mismatch:
    raise SystemExit(f"--budget {args.budget} is IGNORED when --frames-json is "
                     f"given (file holds {sorted(sizes)} frames/item)")
```

Related, same run: budget 0 (feed no video at all, the channel-deletion screen)
crashed on every item with `KeyError: 'pixel_values_videos'`. With no video in
the message the processor emits no video keys, and any bookkeeping that records
`inputs["pixel_values_videos"].shape` must be guarded. The zero-budget arm of a
deletion screen is a genuinely different code path from budget 1.

## A hand-rolled nDCG on MultiVENT reads +28 — the docs are duplicated, the metric decides the ties

Computing nDCG@10 on the CLaMR score matrices with `rank = 1 + (scores > gold).sum()`
gives **85.63** where the official eval prints **57.63**. Same tensor, same
diagonal-gold targets. The strict-greater rank hands the gold every tie it is
involved in — and on MultiVENT 2.0 the ties are not numerical noise: **1495 of
1504 queries have exact cross-doc ties at the gold score, up to 10-way**,
because duplicate channel text across docs (same event → same description/OCR)
produces identical per-channel MaxSim sums. torchmetrics sorts the gold behind
its ties instead: 57.67 on the same matrix. The union/all-token scorer that
looked +28 above the channel-max oracle collapses to 57.54 under the official
metric — that whole "finding" was tie handling.

Two discrepancies stack on these matrices (found independently in reduce_sweep):
tie policy (optimistic vs sorted, ~+28) and a residual ~0.2 between a stable
descending argsort (57.43) and torchmetrics (57.63) — candidates are sort-order
of ties vs binary/graded labels; unresolved, so do not chase 0.2-level deltas
across implementations. `compute_metrics` builds BINARY diagonal targets, so the
official 57.63 is not graded relevance.

Rules: report ONLY torchmetrics `RetrievalNormalizedDCG(top_k=10)` with
compute_metrics' binary diagonal targets on this benchmark; keep any fast
optimistic ranker out of reportable paths (rename it `_UNREPORTABLE`); and the
same duplicated pool makes LP flanking counts unsatisfiable by construction
(gold−distractor margins exactly 0.0), so screen diagnostics must dedup or
margin-epsilon first.

## Auto-selected backends silently change across library versions

The same script, same data, same model can do something different because a
dependency's availability probe answers differently in a new environment.
Instance: qwen-vl-utils picks its video reader by probing
`importlib.util.find_spec("torchcodec")` — which says yes whenever the package
is INSTALLED, even if `import torchcodec` fails (broken .so). Under the new
env the selector silently chose torchcodec over decord, so a subset-reader
hook patched onto the DECORD backend was never called and every
`frame_indices` list was silently ignored — the run would have degraded to
uniform sampling while looking like a selection sweep. The only reason it
failed loudly instead: the torchvision fallback had been deliberately replaced
with a guard, which converted silent corruption into 976 identical errors.
(Also the diagnosis trap inside the trap: the guard's message masks WHICH
upstream stage broke — the first diagnosis blamed a return-contract change
that did not exist. `pip list` agreeing on qwen-vl-utils versions is what
settled it.)

Fixes that hold: set the module-level FORCE variable AND
`get_video_reader_backend.cache_clear()` (it is lru_cached — setting the
global alone does not take if anything already called it), then assert the
chosen backend. Rule: pin backend selection explicitly wherever a library
auto-selects; an availability probe is not an importability proof.

## Composite checkpoints: the sub-model tower must come from the snapshot itself

WAVE-7B finetunes its BEATs audio tower and ships the finetuned weights in the
HF snapshot under the STANDARD filename `BEATs_iter3_plus_AS2M.pt`. Wire
`WAVE_BEATS_WEIGHT_PATH` to any public download of that filename and you get a
base — not finetuned — audio tower: 249/250 tensors differ (max Δ 0.22), zero
warnings, healthy logs, plausible embeddings. Bundled copy vs the main ckpt
agrees to bf16 rounding (max Δ 0.012), which is the fingerprint that the
bundled file is the trained one. Only a tensor-level diff against the snapshot
exposes the swap.

Rule, beyond WAVE: for ANY composite checkpoint (Qwen-Omni-class models bundle
audio towers the same way), sub-model paths never resolve by filename or
env-var to an external download — always the snapshot's own copy, asserted at
load by comparing a few tensors against the main checkpoint. Corollary for
torch ≥ 2.7: the weight_norm shim reports pos_conv `weight_g/weight_v` as
"newly initialized" even when correctly loaded from the bundled file — that
warning is benign ONLY under the snapshot-copy rule.

Two smaller traps from the same runs: FLARE's `eval_run.sh` hardcodes
`CUDA_VISIBLE_DEVICES=${IDX}` (line 221), which REPLACES any outer pin — five
"parallel" arms serialized onto one GPU for 27 h with healthy logs; patch it to
honor an override before multi-GPU use (per-shard resume makes the kill
lossless). And plain-PyPI `torchaudio` 2.11.0 ships a CUDA-13 build that fails
at import (`libcudart.so.13`) beside a cu128 torch — pin `2.11.0+cu128` from
the PyTorch index.

## A guarded fallback reports the guard's message, not the failure

We disabled a library's fallback video reader because it whole-reads hour-long
videos and OOM-kills the job:

```python
def _no_torchvision(ele):
    raise RuntimeError("torchvision fallback disabled: whole-video read OOMs")
VIDEO_READER_BACKENDS["torchvision"] = _no_torchvision
```

The library's `fetch_video` wraps the *chosen* backend in `try/except` and falls
back to torchvision on **any** exception. So every upstream failure — whatever it
actually was — arrives as that one sentence. Two separate root causes produced
byte-identical output across 976 items, twice, and both times the message named
the innocent stage.

The uniformity is what misleads. 976 identical errors reads as "one systematic
problem with my configuration", so you debug the guard. It is really "976 masked
exceptions that happen to funnel through the same handler", and the real cause
was different each time (once a backend-selection change, once a strict-dataclass
validation error four frames deeper).

Make the guard confess what it is standing on:

```python
def _no_torchvision(ele):
    import sys, traceback
    cur = sys.exc_info()[1]
    detail = ("".join(traceback.format_exception_only(type(cur), cur)).strip()
              if cur is not None else "no upstream exception recorded")
    raise RuntimeError(f"torchvision fallback disabled. THE REAL FAILURE IT "
                       f"MASKED: {detail}")
```

That one change turned a two-round diagnosis into a single run. Any time you
replace a fallback with a raise, attach `sys.exc_info()[1]` — you are standing
inside someone else's `except` block whether you meant to be or not.

## Two more ways to run code you did not think you were running

Both cost us real time in the same week, both silent.

**A tracked file copied out-of-band gets reverted by the next copy.** A
collaborator patched a script directly on the cluster; we then edited our local
copy of the *same* file and `scp`'d it over, silently reverting their fix. The
symptom was a bug that had already been fixed reappearing with no commit
explaining it. Rule: files under version control move through the repository
only. If the cluster copy is not a clone, that is the bug.

**A job that invokes a script once per arm can run different code per arm.** A
sweep shaped like `for model in ...; do python eval.py ...; done` re-reads the
script on every iteration. Editing it mid-run means early arms and late arms
execute different versions, inside one results file, with nothing recording the
difference. We got away with it because the edits were verified no-ops, which is
luck. Stage edits and deploy between jobs — and stamp the code/library versions
into the output records so the question is answerable after the fact:

```python
rec["env"] = {"transformers": transformers.__version__,
              "torch": torch.__version__, "model_class": arch}
```

## Moving inputs to the device is not the same as casting them

Three WAVE-7B retrieval arms ran overnight, exited clean, and wrote a complete
769 MB `text.pt` each. Every one had an empty `results/`. 1596 clips failed in one
arm, 399 in each of the others, all with:

```
clip encode failed: Input type (torch.FloatTensor) and weight type (torch.cuda.HalfTensor)
```

The adapter's `_move_inputs` moved tensors and stopped there:

```python
def _move_inputs(self, inputs: dict) -> dict:        # WAVEAdapter
    ...
    elif isinstance(value, torch.Tensor):
        moved[key] = value.to(self.device)           # device only
```

while the generic adapter twenty lines up in the same file does the other half:

```python
value = value.to(self.device)
if torch.is_floating_point(value):
    value = value.to(_wd)                            # _wd = model param dtype
```

What makes this expensive is *which* half of the pipeline it spares. Text
encoding passes, because its tensors are token ids and attention masks —
integers, with no dtype to mismatch. Only the float payload (pixels, waveforms)
hits the fp16 weights. So the run produces a large, correct-looking text side and
a silently empty media side, and the arm that looks furthest along is the one
that has computed nothing.

Two rules. Any `to(device)` on model inputs needs a dtype clause derived from the
model itself — `next(self.model.parameters()).dtype` — never a hardcoded half.
And a tensor left on the CPU shows up as `torch.FloatTensor`, not
`torch.cuda.FloatTensor`; that prefix tells you whether you are looking at a
missed cast or a missed move. Here it was a key deliberately passed through
unmoved, which the error named precisely if you read the type and not just the
mismatch.

The second half of the failure is the one that reaches a table. Each clip died
inside a per-item `except` that appended to `errors.txt` and continued, so the
job exited 0. If anything downstream scores a missing gallery entry as "not
retrieved" rather than "not computed", an empty run becomes a *low number*
instead of a crash — and a depressed row against a baseline you are trying to
beat is exactly the result nobody double-checks. Assert the count before scoring,
in the scoring code, not the encoder:

```python
assert len(results) == len(gallery_ids), \
    f"{len(gallery_ids) - len(results)} items never encoded — refusing to score"
```

Companion to *A guarded fallback reports the guard's message*: that one masks
which failure happened, this one masks that anything failed at all.

## An orphaned job outlives the ssh session that launched it

A VPN drop killed the ssh sessions driving mm-lab01. One eval process kept
running with `ppid 1`, adopted by init; from outside it looked as dead as the
terminal that started it. Someone relaunched. Fourteen hours later two processes
were walking the same video list into the same output directory with
byte-identical arguments — same `--run_name`, same `--exp_dir`, both
`--num_chunks 1 --chunk_idx 0`, so no sharding kept them apart.

The obvious worry is a torn `.pt` — concurrent writers both passing a
skip-if-exists check, producing the kind of file that loads without complaint.
We audited it rather than assuming it: `torch.load` over all 164 `.pt` files in
the contested directory gave **0 load failures and a uniform (1024,) shape**,
none under 10 kB, after 16 hours of overlap. The write window per file is
milliseconds, so the race is real in principle and did not fire once here. Carry
the raw counts with the claim; without them this entry reads as a corruption
finding when what we have is a clean audit.

The damage is elsewhere, and it is total rather than partial. Two processes
walking the *same list in the same order* under skip-if-exists do not split the
work: the later one skips everything already on disk, catches up to wherever the
first has reached, and then shadows it — both computing the same item at the same
time, forever, because neither can see output the other has not written yet. A
duplicate does not halve the remaining time. It contributes nothing, on a shared
box where someone else is queued.

Every connection drop manufactures one of these, so the guard belongs in the
launcher rather than in anyone's memory:

```bash
pgrep -f "run_name $ARM" >/dev/null && { echo "$ARM already running"; exit 0; }
```

Two diagnostic corrections came out of it. **Age does not identify the
duplicate.** The instinct is to kill the younger process; here the younger one
came from the *newer, fixed* launcher (`eval_run_pinned.sh`, Aug 12) while the
survivor was started by the older script, so killing by recency would have kept
the stale run and destroyed the good one. Provenance comes from `ppid` and the
launcher, never from start time. **And `ps` truncates the command line** before
the argument that names the arm, so the whole duplication is invisible in `ps`
output — the identity is in `/proc/PID/cmdline`, NUL-separated:

```bash
tr '\0' ' ' < /proc/$PID/cmdline; echo
```

## A flag accepted without error is not a flag honored

A session was launched with `--remote-control` in print mode, reported nothing
unusual, and then failed to appear as a peer. That absence was read as evidence
the feature did not do what we thought. It wasn't: print mode ignores the flag
outright — no connection, no warning, no non-zero exit. Driven from a real TTY
the same binary announced "/remote-control is active".

The cost is not the wasted launch, it is the false negative it produced. A
feature that silently no-ops returns exactly the observation you would get from a
feature that works and disproves your hypothesis, and a null result is the thing
we are least inclined to re-test. Before you believe a negative, confirm the
feature *announced itself* — a banner, a log line, a status call — not merely
that the process started and the flag was accepted.

**The same shape cost us a GPU-week, and here it is measurable.** A model was
configured with `attn_implementation="sdpa"` and OOM'd on a single 47 GiB
allocation. The first diagnosis was that passing an attention mask drops SDPA
onto the math backend; the real precondition is narrower and stricter. Qwen's
vision and audio attention pass **3-D** tensors `(heads, seq, dim)`, and every
fused SDPA kernel requires 4-D — so they silently refuse, and *every* backend
falls back to MATH regardless of the mask. "sdpa is set" was never "sdpa is
used", for the whole run.

Forcing torch's backends against each other on the same inputs shows they are
selectable in principle and identical in practice at these shapes:

```
MATH vs EFFICIENT vs FLASH vs CUDNN:  max|Δ| = 9.766e-04 for every pair
                                      = 2^-10 exactly, one bf16 ULP
```

Two consequences. The memory fix that worked (block-diagonal attention over
`cu_seqlens` segments with `attn_mask=None`, 47 GiB → 19 GiB) worked by shrinking
per-call `n` inside the math backend, **not** by enabling a fused kernel — which
is why the result is 19 GB rather than something far smaller, and reshaping to
4-D is still on the table as a separate win. And a naive version of this
backend-comparison test reports "no available kernel" and yields nothing, which
reads like the test failing rather than like the finding it is.

Check which backend actually ran before attributing anything — memory,
throughput, or numerics — to the one you asked for.

## Two implementations agreeing is not two findings

We reported a tuned fusion weight as confirmed, because two lines found the same
optimum: w\*=0.95 worth 12.81, from a held-out α sweep and from an independent
scalar sweep, "matching to the decimal". It went into a publication entry as a
double-find.

Neither grid contained 0.93. A 101-point grid, tuned on train folds and scored
held out, picks **0.93 in all five folds**:

```
0.90:12.80   0.93:12.86   0.95:12.81   0.97:12.73   1.00:12.61
```

Both sweeps were forced to report the same wrong answer, and their agreement
measured the shared grid rather than the data. This is a distinct failure from
two analyses sharing cached scores: the code can be genuinely separate, the
splits genuinely different, and the agreement still manufactured — by any
constraint both inherited. Search space, tolerance, candidate list, tokenizer,
seed set.

Before citing agreement as evidence, ask what the two runs shared. If the answer
includes the set the answer had to come from, you have one measurement.

The correction also shows the other side. The plateau here spans 0.06 from 0.90
to 0.95, which is *larger* than the 0.05 error being corrected — so the right
report is an interval over the plateau, not a second decimal. A number too
precise to be stable invites exactly the re-derivation that produced this entry.

## A surrogate that disagrees with the reported metric hands the flexible arm a win

Testing whether a per-query weight beats a global one, the conditional arm won by
+0.41. It was an artifact of the training objective. Listwise cross-entropy put
its optimum at w=0.84 — worth 12.24 under the top-1 metric actually being
reported — while the true top-1 optimum was 0.93 at 12.86. The constant-weight
control was therefore fitted to the wrong target and ran handicapped, and the
flexible arm's extra capacity let it partially escape the same mis-specification.
The measured "gain" was the control's handicap.

Any capacity comparison needs both arms optimized for the metric in the table.
When they cannot be (non-differentiable objectives usually), certify the harness
first: check the SIMPLE arm recovers its known optimum. Here the constant arm
reaching 12.84 against a grid optimum of 12.86 is what licenses believing
anything about the conditional arm — without that check, every number downstream
is measuring the surrogate.

The general shape: extra capacity buys escape from a badly specified objective,
and that escape is indistinguishable from the effect you are trying to measure.

## Three identity checks that confidently return the wrong answer

All three fired in one afternoon, on the same investigation, and each one flipped
a conclusion.

**Hash the payload, not the container.** Seven retrieval arms wrote per-video
`.pt` files; `md5sum` gave seven different hashes, which reads as seven distinct
computations. Loading the tensors said the opposite — arms sharing `media_mode`
were **bit-identical**, `max|diff| = 0.00e+00`. The files differed only because
the dict stores `query_mode` and other metadata beside the payload:

```python
{'video_id':…, 'model':…, 'query_mode':…, 'media_mode':…, 'clip_ids':…, 'clip_feat':…}
```

`query_mode` is recorded and never touches `clip_feat`. Four of the seven arms
were recomputing another arm's numbers — ~57% of a 17-hour encode — and the
cheap check said everything was fine. Whenever metadata shares a serialized
object with data, compare the tensor.

**A comparison over a vanished process reads as agreement.** Checking whether two
processes had identical arguments:

```bash
diff <(tr '\0' '\n' < /proc/$A/cmdline) <(tr '\0' '\n' < /proc/$B/cmdline) && echo "IDENTICAL"
```

Both processes had already exited. Both reads produced empty output, `diff`
found no difference, and it printed `IDENTICAL` — a positive claim manufactured
out of two absences. Assert both streams are non-empty before comparing
anything that can disappear.

**`pgrep -f` matches the command that runs it.** Searching for live workers:

```bash
pgrep -u "$USER" -f complete_retrieval_pipeline     # finds your own shell
```

The pattern appears in the searching shell's own command line, so a dead
pipeline reports one live process. That phantom was briefly read as evidence
that work had restarted and was regenerating duplicates. Use the bracket trick
so the pattern cannot match itself, and confirm against something independent —
GPU compute apps, or output counts that should be advancing:

```bash
pgrep -u "$USER" -f '[c]omplete_retrieval_pipeline'
```

**And the bracket trick is not sufficient on its own** — this is the fourth
instance, and the most instructive, because here the *mitigation* failed toward
the confident answer. Verifying the phantom, the second line's bracketed pattern
still returned 1:

```bash
pgrep -u "$USER" -f complete_retrieval_pipeline   | wc -l   # self-matching
pgrep -u "$USER" -f '[c]omplete_retrieval_pipeline' | wc -l # "safe" — also 1
```

The idiom keeps the literal string out of the *pattern*, not out of the
**command line**. Running both checks in one invocation — or adding an
`echo "checking complete_retrieval_pipeline"` label — puts the unbracketed
string back into the searching shell's own cmdline, where the bracketed pattern
happily matches it. The target string must appear nowhere else in the
invocation: one check per command, no echo labels naming the target.

Then pass `-a`, so a phantom prints its own command line instead of a bare
count. A count cannot distinguish "found a worker" from "found myself"; a
listing shows the ssh line immediately. The same tell caught it from the other
side — a phantom pid with `etime 00:00` carrying `--encode_gallery`,
`--encode_text_chunk`, `--merge_gallery` and `--calc_metrics` simultaneously,
which no real worker ever does.

Best of all, cross-check liveness against whether the **outputs are advancing**.
Result counts identical across two readings minutes apart settles it
independently of any process listing, and would have been right in every one of
the four cases above.

The same rule protects against the opposite error, which cost another line two
relaunches: a job that has produced *nothing yet* is not necessarily hung. An
accidentally quadratic setup step — a dict comprehension nested inside a list
comprehension, rebuilding a 37,029-entry map for each of 53,580 queries — sits
silent for many minutes before its first print, which is indistinguishable from
a bad mount. Before killing a silent job, check whether it is burning CPU.

**The same trap in `pkill` does not return a wrong answer — it kills you.**
Clearing the redundant workers:

```bash
pkill -u "$USER" -f "exps_mmlab"      # matches the ssh command running it
```

The remote command's own line contains `exps_mmlab`, so `pkill` terminates the
session mid-script, leaving the actual cleanup half-done and the operator
guessing which processes died. Every rule above applies, and the stakes are
higher because the failure is destructive rather than merely confident:

```bash
pkill -u "$USER" -f "exps_mml[a]b"
```

Never let a `pkill` pattern be the first thing you test. Run it as `pgrep -a`
first and read the list — if your own shell is in it, the `pkill` would have
killed you.

The family resemblance is worth more than the five instances: each check
answered a question about *identity* — same bytes? same process? same run? — by
consulting a proxy that was cheap to read, and each proxy failed toward the
confident answer rather than toward an error. Including, twice, the proxy
brought in to check the previous proxy.

## Overlapping marginal CIs are not evidence of no difference

A tuned fusion weight scored 12.86 with a bootstrap CI of [12.59, 13.15]. The
baseline it had to beat — drop the second channel entirely — sits at 12.61,
comfortably inside that interval. Read as "indistinguishable", the tuning result
would have been retracted.

Both arms score the **same 53,580 queries**, so almost all of the marginal width
is shared query difficulty, which cancels in the difference. The paired
bootstrap over the per-query difference gives **+0.254, CI [+0.110, +0.394]** —
excludes zero. The comparison that overlapped by 0.5 points resolves at 0.14.

Whenever two arms consume identical items, the marginal intervals answer a
question nobody asked ("how well is this arm known in absolute terms") and are
systematically too wide for the question that matters. Bootstrap the paired
difference and quote that interval; the same applies to seed-to-seed spread,
which is why a paired design beats two independent runs.

The converse trap sits one step later: **a difference measured at the argmax of
the same data inherits the selection.** Quoting the peak's advantage over a
baseline is optimistically biased even when the tuning was fold-wise. When the
curve is a plateau, report the plateau's MINIMUM advantage — here "every w in
[0.90, 0.95] beats the baseline by +0.19 to +0.25" — which costs nothing, is
selection-free, and still excludes zero.

## Per-item resume turns a settings change into a two-protocol gallery

A run was encoding video at 32 frames. The frame budget was wrong — the upstream
default is 64, and the 32 was a workaround for an OOM that turned out to be an
attention-kernel bug, not a capacity limit. Two videos per arm had already been
encoded when the setting was corrected.

The pipeline resumes by skipping items whose output already exists. That is the
right design for surviving node flaps and it is exactly what makes this
invisible: the finished gallery would have contained two videos at 32 frames and
397 at 64, with nothing in the output recording the seam and no error anywhere.
Every downstream metric would have been computed over a mixture of protocols.

**A settings change invalidates the cache.** Delete the affected outputs, or
version the output directory by config. Per-item resume is what removes your one
chance to notice, because the run that mixes protocols looks exactly like the run
that resumed cleanly.

Two habits that contain it:

- Never write a settings probe into the real output directory. Verify the new
  configuration under its own `RUN_NAME`, then move that directory out of the
  tree entirely so no later glob can score it.
- Stamp the settings into each record (see *A job that invokes a script once per
  arm*). A mixed gallery is then detectable after the fact instead of permanent.

## A number you derived becomes a number you believe

An operator reported a run as `67/738` and planned an eight-hour wait around it.
The gallery held **399** videos: `dataset_summary.json` said
`total_videos: 399`, and the four workers logged `Processing 100 videos in chunk
0/4` — 100+100+100+99. The `738` had been back-derived as 2 × 369 from a chunk
count in an earlier smoke test, for a different purpose, and then carried
forward as though it had been measured.

Nothing about it looked like a guess by the time it was being used. That is the
whole hazard: a figure computed once for one purpose acquires the authority of an
observation, and the arithmetic that produced it is not attached to it.

The damage here would not have been the wrong ETA. A run that stops at 399
against an expectation of 738 reads as a **silent early exit** — the same shape
as the empty gallery two entries up, and it would have been "fixed" by
relaunching a job that had already finished.

Before a number governs a decision, say where it came from out loud. "Measured
from `dataset_summary.json`" and "derived from a chunk count in a different run"
should never be indistinguishable, and progress counters are a common source of
the confusion: the `n/n` inside a per-worker progress bar was counting *clips
within the current video*, not videos.


## A bootstrap with the hyperparameter held fixed is too narrow

Tuning a fusion weight and then bootstrapping the result, we resampled queries
while holding the tuned `w` at its full-data value. The honest procedure redoes
the selection inside every resample, because the thing being estimated is
"tune on your data, apply to new data" — and the tuning is part of it.

  weights fixed, queries resampled     +0.254   95% CI [+0.112, +0.398]
  selection redone inside each resample +0.254   95% CI [+0.063, +0.386]

**The point estimate did not move at all**, and this is what makes the trap hard
to see. Every one of the 5 folds selects the same `w`, so off-fold selection
returns the plug-in answer and nothing looks wrong. The bias was never in the
estimate — it was in the **interval**, which was 1.13× too narrow, moving the
lower bound from +0.112 to +0.063. We had been chasing a bias in the wrong
place: the prediction going in was that the estimate would drop to ~+0.20, and
it did not budge.

Selection is stable on the data you have and unstable under resampling — the
same grid whose argmax is unanimous across folds is the argmax in only 79.8% of
bootstrap resamples. A fixed-weight interval quietly asserts that the tuning
step contributes no variance.

Any interval around a tuned quantity has to include the tuning. The tell that
you have it wrong is comfort: an interval that does not widen when you add a
selection step is an interval that never modelled the selection.
## A tolerance built from an unexplained discrepancy is not a tolerance

An arm was gated against a published anchor of 7.58. We set the pass band at
7.0–7.6, reasoning that another reproduction on the same benchmark ran 2–8%
below its published values, so this one probably would too. The arm returned
**7.39** — inside the band, and wrong. It had been fed 32 seconds of audio where
the protocol wants 64.

The band was built out of a discrepancy nobody had explained. That is the whole
error: an unexplained offset is not a known noise level, and using it to widen a
gate converts the gate into a device that accepts the thing it was built to
catch. Every additional unexplained result makes the band wider and the gate
weaker, which is exactly backwards.

Two rules that survive it:

- **A tolerance comes from the expected size of harness noise** — decode
  nondeterminism, seed spread, tie handling — not from the size of an
  unexplained gap you are already carrying.
- **A systematic same-signed pattern is a failure at any magnitude.** 7.39 vs
  7.58 undershoots at R@1, R@5 and R@10 together, 2–4% each. Noise does not
  agree on a direction three times. A genuinely broken tower, for contrast, reads
  like 1.65 against 12.87 — the near-miss is the informative case, and the only
  one a published anchor can catch.

Corollary, from the same investigation: check whether the discrepancy is
*uniform* before assuming it cancels. The same reproduction was −2.0% on
vision-only media, −7.7% fused, and −19.4% audio-only — monotone in audio
content, i.e. an audio-path deficit rather than a level shift. "It shifts every
arm together, so orderings are preserved" was the natural assumption and its own
numbers refuted it. A discrepancy correlated with the axis your headline varies
along is a bias in the headline, not a constant.

## Measure your control's floor on data you know is identical

An integrity check reported "min per-query cosine between arms = 0.999998" and
was read, by two people, as *nearly* identical — from which we inferred about
1e-6 of run-to-run nondeterminism in the text tower, and derived a rule that
differences at the third decimal were not meaningful.

There is no nondeterminism. The arms are **bitwise identical**, `torch.equal`
True, `max|Δ| = 0.000e+00`. Compared against a byte-for-byte copy of itself, the
same check still returns **0.99999827** — that number is the FLOOR of
`cosine_similarity` on fp32 embeddings of width 3584, not a measured difference.

So the control was unfalsifiable in the range that mattered. It returns ~0.999998
for identical data, which means it cannot separate "same" from "differs by 1e-6",
and a real 5e-6 discrepancy would have been dismissed as noise *by the rule we
derived from its floor*. A check that cannot fail in the regime you are screening
is not evidence, and it is worse than no check because it produces a number that
reads like reassurance.

Two habits:

- **Calibrate the control on a known-identical pair first.** Run it on a copy.
  Whatever it returns is your floor, and your detection threshold has to sit well
  above it. This costs one line and converts an opinion into a measurement.
- **Report `max|Δ|` and exact equality alongside any similarity.** Their floor is
  exactly zero, so they stay informative where cosine saturates. Reserve cosine
  for the regime it is good at — catching gross divergence, not certifying
  identity.

The same blind spot lives in every "cosine ≈ 1.0 so the mirror is intact" check.
Ours read on embeddings of width 3584; the floor moves with dtype and dimension,
which is exactly why it has to be measured rather than assumed.

## Audit the object, not just the estimate

The most expensive error in this file was not caught by any of the checks above,
and it survived a full day during which two lines audited the result repeatedly.

The claim was "no learned per-query operator beats a well-chosen fixed one" —
measured at −0.349 with a paired CI excluding zero, on a head initialized at the
best constant so that standing still was free. Over that day it was checked for
selection bias, for grid resolution, for seed stability, for fold-partition
stability, for objective/metric mismatch, for multiplicity, and for whether a
per-group weight could reach what the per-query head could not. Every one of
those audits was about **the estimate**. None of them asked what was being
estimated.

The sweep was running `w·(q·v) + (1−w)·(q·a)`. The incumbent operator is
`q · normalize(w·v + (1−w)·a)`. Re-measured on the family the benchmark actually
uses, the same code with the same folds gives **+0.159** — significantly
*better*, not worse. The headline null was an artifact of a modelling choice
nobody had written down, and no amount of rigour applied to the number could
have found it, because the number was computed correctly throughout.

Two habits, and the first is nearly free:

- Before auditing a result, **write down the object**: the exact formula the
  baseline computes, the exact population the metric averages over, the exact
  set the max ranges over. One line, in the note, next to the number. Most
  family errors die on contact with being written out.
- Notice that **statistical rigour is not a defence against this**. Confidence
  intervals, nested selection and partition checks all condition on the object
  being right; they get sharper as the object gets wronger, and a tight CI on the
  wrong quantity is more persuasive than a loose one.

The tell, in hindsight, was endpoint agreement (see *Sweep the operator the
benchmark actually uses*) — but the general lesson is broader than fusion
operators. Any time a comparison is embedded in a parameterization, the
parameterization is a claim, and it is the one claim nobody thinks to check.

Corollary observed on the same benchmark, four times: a single-configuration
value reads high. Single split +0.192 → five-partition mean +0.159; the first
grid's argmax, the first fold's optimum and the first bootstrap's interval all
went the same way. Whatever you measured once, measure under a second
configuration before quoting it.

## The aggregate is more stable than the retrieval it summarizes

Chasing a 0.19-point gap against a published anchor, we asked how much of it a
pure numeric difference could produce. Same embeddings, same gallery, same
20,000 queries; the only change was running the score matmul in bf16 instead of
fp32:

```
R@1 fp32 = 7.4200
R@1 bf16 = 7.3150      delta = -0.105
top-1 retrieved item CHANGES for 10.1% of queries
```

Ten percent of the retrieval churns and the metric moves by a tenth of a point,
because most flips swap two distractors and never touch the gold. Two things
follow, and they point in opposite directions.

**R@1 is robust, and comparisons across configurations still are not.** The
margin distribution explains both: rank1 − rank2 has p1 = 4.8e-5, p5 = 2.3e-4,
p25 = 1.4e-3, so a perturbation of ε flips roughly the queries with margin below
ε. At ε ≈ 1e-3 that is 19% of the population, against effects of interest worth
0.19–0.47 points — 100 to 250 queries out of 53,580. The at-risk population is an
order of magnitude larger than the effect. So any comparison against a published
number, or across boxes, kernels, precisions or a library upgrade, carries a
floor of this size, and a harness gate built on it can catch a truncated input
but cannot resolve tenths.

**Within-run contrasts are protected, and the warning must not be over-applied.**
Arms that share embeddings and differ only in a swept parameter see the
perturbation as common-mode: it moves both arms nearly identically and cancels in
the difference. Same-config differences stay interpretable at 0.1 points; levels
compared across configurations do not.

**Per-item artefacts are an order of magnitude more fragile than the metric over
them.** Oracle picks, per-query win/loss tables, flip sets and the qualitative
examples that go in a paper all live at the item level, where 10% churn is the
operative number rather than 0.1 points. Source them from the same run they are
argued about, never from a re-run.

**The floor is a rank-1 phenomenon — do not carry it to deeper cutoffs.**
Measuring the same perturbation at each cutoff, 20,000 queries per arm:

```
           ΔR@1     ΔR@5     ΔR@10
  vision   -0.095   -0.040   +0.055
  audio    -0.105   +0.040   +0.050
```

It shrinks with K and is sign-unstable by R@10, because falling out of a top-10
set takes far more than falling out of rank 1. Two consequences. The cost is
**level-independent** — 0.095 on a 16.7-point row against 0.105 on a 7.4-point
row — so equal absolute gaps across rows of very different strength are the
numeric signature rather than a coincidence; the larger gold-near-top population
on the stronger row is offset by its larger margins. And "within the numeric
floor" is a claim available only at R@1. In the case that produced this table,
R@1 sat about two floor-units low and looked explainable, while R@5 and R@10 sat
10-20x the floor — so a deficit that reads as reproducible at rank 1 was a real
systematic gap at depth. Check every cutoff before writing "reproduces".

⚠️ One trap in doing this: a bf16 matmul establishes the transfer function
(ε → ΔR@1), not the size of the perturbation you are attributing to. Reading
"numeric noise explains my gap" off it requires knowing your actual ε, and
kernel-vs-kernel differences can be anywhere from 1e-5 to 1e-3 — which map to
R@1 movements of 0.005 and 0.105 respectively. Measure ε directly (re-encode with
a different batch composition; accumulation order changes, nothing else does) and
read the predicted churn off the margin CDF.

## Before asking whether a setting is correct, ask whether it is yours

Chasing a reproduction gap against published numbers, we spent an afternoon on
four candidate causes: a pixel cap, a frame budget, per-modality decode defaults,
and the query prompt template. Three died the same way, and the shared argument
is cheaper than any of the investigations that found it.

If a value is **upstream** — the benchmark's own code, unmodified — then the
authors' run used it too. It is common-mode with the anchor and cannot produce a
discrepancy, whatever its value, and whether or not it looks sensible. The
question "is `clips_per_video=3` too small?" is unanswerable and irrelevant; the
question "did *we* set it?" is a `git diff` and settles it.

```bash
git -C "$REPO" diff --stat        # if your change isn't listed, it isn't yours
```

Run that first. It eliminates most candidates in seconds, and it eliminates them
on a stronger argument than plausibility ever gives you — the surviving hunt is
then confined to what you actually changed plus what is genuinely outside the
repo (inputs you extracted, hardware, library versions).

⚠️ **Necessary, not sufficient: "upstream now" is only "upstream then" if your
clone is at the commit that produced the published numbers.** A repository that
moved after publication carries defaults the authors never ran, and they will
still show up as untouched in your diff. Pin the check:

```bash
git -C "$REPO" log -1 --format='%H %ci'          # what am I actually at?
git -C "$REPO" log -S'176400' -- path/to/file    # when did this value appear?
```

If the value postdates the paper, it is upstream AND a deviation — the one
combination the quick test reads as safe.

## A protocol fix can be correct and worth nothing

A run was found sampling 32 seconds of audio where the benchmark's own default is
64. It was a genuine deviation from protocol, it was fixed, and the fix was
described — by the line that found it and by the critic who relayed it — as the
save of the day.

Both budgets were then measured on the same 399-video gallery:

```
        R@1      R@5      R@10
32s   7.3908   17.831   23.888
64s   7.4057   17.816   23.998
Δ     +0.015   −0.015   +0.110
```

Nothing. And it was predictable in advance from the data: median clip audio is
6.8s, p99 is 42.7s, and the code path *subsamples* rather than truncates, so a
32-second budget touched 2.5% of clips and only thinned them.

Two habits fall out. **Fix protocol deviations regardless of expected size** —
compliance is not conditional on the effect, and re-running converted a live
confound into a measured control that closes the question permanently. But
**attach a magnitude estimate before calling a fix important**, and derive it
from the data distribution rather than from the fact that a bug was real. One
line — what fraction of items does this touch, and by how much — would have
stopped the overclaim at the moment it was made rather than three retractions
later.

The general form: "there was a bug and we fixed it" says nothing about whether
any number moved. Those are separate claims and the second one needs its own
measurement.

## Four failures in one day, one shape: measuring a proxy that cannot express the state you care about

Each of these was found and fixed separately. They are one entry because the
fix for each is the same question, and asking it costs nothing.

**Cosine on identical data.** A control compared two embedding sets and reported
min per-query cosine `0.999998`, read as ~1e-6 run-to-run nondeterminism. The two
tensors were **bitwise identical** — `torch.equal` true — and
`cosine_similarity` still returns `0.99999827` on fp32 embeddings of width 3584
with row norms in `0.999999–1.000001`. That value is the comparator's floor, not
a measurement. The control could not fail in the regime it was screening, and
the tolerance derived from it would have dismissed a genuine 5e-6 difference as
noise.

**`pgrep` matching its own command line.** A wait-loop polled
`pgrep -f complete_retrieval` until zero. The polling shell's own `bash -c`
cmdline contained that string, so the count never reached zero and the loop
waited its full timeout while the GPUs sat idle. Bracket the pattern
(`[c]omplete_retrieval`) *and* keep the literal string out of the same command.

**A liveness check watching the wrapper.** Killing runs by wrapper name left the
python workers orphaned, because the workers' cmdlines do not contain the
wrapper's name; and checking the wrapper's liveness reported "healthy" for runs
whose workers had already died. Verify the process that does the work, and
verify progress (shard counts growing) rather than existence.

**Free GPU memory read as ownership.** A GPU map was inferred from
`memory.used`, concluding four cards were free and one was idle. The map was
exactly inverted: those cards held four live workers, and the "idle" one was
between clips at 0% utilisation. Broadcast to three peers, it would have put a
new job on top of a running arm — and co-residency on those same cards had
already cost 14 videos to CUDA OOM on allocations as small as 128 MiB. Query by
**process owner**, not by free memory or utilisation.

The common form: each check substituted something cheap and observable for the
property actually in question — similarity for identity, a process name for a
process, a name for the work it does, free memory for ownership. Three of the
four could not have returned the answer that would have falsified them.

**The habit that fixes all four in one line: calibrate the check on a case whose
answer you already know.** Run the identity check on a byte-for-byte copy. Run
the pgrep with nothing running. Ask what the liveness check says about a process
you just killed. Ask what the GPU query says about a card you know is yours. If
the check cannot fail on the known case, it is not measuring what you think, and
you have learned that for the cost of one command instead of one wrong
conclusion.
