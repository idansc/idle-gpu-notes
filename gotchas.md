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
they reproduce published LongVideoBench numbers (58.3 @64 frames), which is
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
margin-epsilon first. Note 04's pool audit already measured the cause from the
data side — a median of 3 exact duplicates of the gold per query pool: exact
duplicates ⇒ identical channel maxes ⇒ exact ties at gold. One cause, two
symptoms (04's LP degeneracy; the 1495/1504 tie rate here).
