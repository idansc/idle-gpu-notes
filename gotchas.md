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

The family resemblance is worth more than the three instances: each check
answered a question about *identity* — same bytes? same process? same run? — by
consulting a proxy that was cheap to read, and each proxy failed toward the
confident answer rather than toward an error.
