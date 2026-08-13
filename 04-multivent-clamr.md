# 04 — MultiVENT 2.0 / CLaMR: finding a benchmark with genuinely multimodal items

> ## ⚠️ THE LADDER DROPPED A REAL CHANNEL — every number below is under revision
>
> **The sweep computes over 4 of the 5 slices, and the dropped one is not
> padding.** `reduce_sweep.py:32` carries an explicit, load-bearing assumption:
>
> ```python
> # channel 0 is padding (M = max(modality_id)+1, ids are 1..4); drop it.
> sims = sims[..., 1:]
> ```
>
> That comment is wrong. Measured on `sims[..., 0]` by the critic line:
>
> | | |
> |---|---|
> | min / mean / max | +6.47 / +10.79 / +22.01 — finite, same scale as the other four |
> | R@1 using slice 0 **alone** | **20.15%**, against chance of 0.066% (1/1504) |
> | tokens with `mod_id == 0` | 1,280,646 = **47.3%** of all doc tokens, 851.5/doc, in 99.4% of docs |
> | `torch.unique(mod_ids)` | `[-100, 0, 1, 2, 3, 4]` |
>
> **Padding does not score 300× chance.** And `-100` appearing alongside `0`
> settles the id question: `-100` is the ignore value, which leaves `0` as a
> content id. Slice 0 is the *second*-strongest single channel in the tensor.
>
> Note the score-level evidence stands independently of whether slice *k* holds
> the tokens whose `mod_id` is *k*. Whatever slice 0 is, it carries strong signal
> and the sweep discards it; the correspondence question decides what to *call*
> the channel, not whether dropping it was safe.
>
> **What is not yet known is the cost.** The comparison that quantifies it —
> max over slices 1–4 (what this ladder actually feeds) versus max over all five
> — has not been run; both lines lost cluster access mid-check. Published
> neighbours for scale, on a different metric so not directly comparable to
> 57.43: max over 0–3 = 21.54, max over all five = 24.14.
>
> **Not everything below is equally at risk, and which is which is decidable on
> paper.** Over-flagging costs something too: a note where every number is
> provisional tells a reader nothing about what to trust.
>
> **Safe in direction — these are LOWER BOUNDS and cannot be overturned:**
> - **LP reachability 89.3%.** The 5-channel feasible set strictly contains the
>   4-channel one (set `w₀ = 0` to recover it), so a query that was reachable
>   stays reachable. Adding slice 0 can only raise it. Keep asserting it, as a
>   lower bound.
>
> **At risk, can move either way:**
> - **max 57.43.** Maxing over *more* channels is not monotone in a ranking
>   metric, and there is a direct measurement of it hurting: on an ad-hoc R@1,
>   slice 4 alone scores 24.47 while max over all five scores 24.14. A
>   weak-but-confident channel wins pairs it should not.
> - **learned `w(q)` 56.23.** More inputs, more capacity, more to overfit.
>   Direction unknown.
> - **the reproduction, 57.63 vs published 58.47** — the number most at risk,
>   because it inherits max's non-monotonicity *and* faces outward rather than
>   comparing arms within one tensor.
>
> **Neither — 59.12 was already retracted and monotonicity does not rescue it.**
> The containment argument applies to the *exact* per-query optimum. 59.12 is the
> **64-sample Dirichlet** estimate, and 64 draws sample a 5-simplex *worse* than
> a 4-simplex, so the estimate can fall while the true optimum rises. This is the
> same failure this note already documents: a sampled quantity is not a bound.
> Do not quote 59.12 in either direction.
>
> ### Pre-registered, before the route returns
>
> Our reproduction sits **0.84 below** published. Restoring slice 0 should land
> the max arm in **[58.0, 58.9]**. All three outcomes are informative:
> - **inside** → the dropped channel explains the reproduction gap and closes it;
> - **below 58.0** → the channel was not the cause and the gap is something else;
> - **above 58.9** → we now over-reproduce, and something is wrong in the other
>   direction.
>
> Written down now so the result is a test rather than a description.
>
> One correction to how this was first reported to me: the at-risk slice is index
> **0**, the one the code drops — not the "slice 4" whose strength prompted the
> question. Slice 4 was inside the four all along, so findings about slice 4's
> strength describe this ladder's *input*, not something it excluded.
>
> **Tie convention**, since it is the other free parameter in play: this ladder
> does not use torchmetrics. `reduce_sweep.py` ranks with
> `torch.argsort(..., descending=True, stable=True)` and locates the gold by its
> position, so ties resolve by **index order** — neither gold-first nor
> gold-last. Deliberate, and documented in the file: the optimistic
> `(score > gold).sum()` convention counts ties as not-ahead and inflated max to
> **85.63** against the true **57.63**. Any arm compared against these numbers
> must call the same ranker, or the tie convention becomes a free parameter
> worth more than the effect being chased.


**Status: reproduced at 57.63 (published 58.47), then abandoned. Every operator
avenue here is bounded below +2 points, and we can now show that cheaply.**

The post-mortem in Finding 4 is the valuable part — it took six failed runs to
get there, and every failure was an environment problem that produced a
plausible number rather than a crash.

## The question that led here

Fusion operators need **several utilities per item** to have anything to do.
[03](03-mmeb-visdial.md) failed partly because each item was one image.

So: does a retrieval benchmark exist where a **single item carries several
modalities at once**, as opposed to a benchmark that merely *covers* many
modalities across separate single-modality tasks?

## Finding 1 — most "omni" benchmarks are unions of single-modality tasks

This is worth stating loudly because it is easy to assume otherwise from the
paper abstracts.

- **MMEB-V3 / OmniSET** groups semantically equivalent instances into
  `{text, image, video, audio}` tuples and then expands each tuple into **12
  directional single-modality-in / single-modality-out retrieval tasks**. Four
  parallel renderings of one meaning; never combined into one input. Its video is
  synthesized and its audio is TTS from captions, so there is not one real
  soundtrack in it. As distributed it is also small (~200 rows in the copy I
  pulled).
- **MMEB-V2** as distributed ships **sampled frames only**; the raw videos have
  been "to be released" for a long time, and a recursive file listing for
  "audio" returns nothing.
- **MIEB** is image+text only. **MAEB** is audio-only items.

Sharpest evidence that these benchmarks do not reward joint reduction: on
MMEB-V3 the top model (Qwen3-VL-Embedding-8B, 49.9) **has no audio encoder at
all**, while audio-capable omni models trail it, and WAVE-7B — arguably the most
genuinely joint audio-visual model — finishes last at 26.3.

> If you are looking for a testbed for multimodal *fusion*, checking that items
> are multimodal (not just that the benchmark is) will save you weeks.

## Finding 2 — MultiVENT 2.0 is real

One video carries four separately-queryable channels: keyframes, Whisper ASR of
speech, OCR of on-screen text, and a metadata description. Query IDs encode which
channel they target (`..._base_0`, `..._speech_0`, `..._ocr_0`, `..._description_0`).

Verified from the judgments rather than the paper: of 417 annotated videos, **194
are gold for all four query types and 164 more for three** — 86% carry ≥3
co-present utilities.

The incumbent method, [CLaMR](https://arxiv.org/abs/2506.06144), publishes the
operator gap on fixed backbone and data:

| | nDCG@10 |
|---|---|
| Qwen2.5-VL, standard contrastive + last-token pooling | 52.2 |
| CLaMR late interaction | **58.47** |
| CLaMR with modalities encoded separately (no cross-modal contextualization) | 44.53 |

That 44.53 → 58.47 gap (+13.94) is what letting utilities interact is worth,
with the reduction held fixed. Their ablation is honest: `combine_modalities`
only changes whether the channels go through the LM together, not how they are
scored.

**And CLaMR ignores audio entirely** — the shards ship `.m4a` tracks, and their
own codebase contains a `ColQwenOmni` variant listing
`["video","image","ocr","asr","description","audio"]` that the released model
does not use.

## Finding 3 — reconstructing their eval set without their Drive

Their processed data link (Google Drive) returns 404. It is fully reconstructable:

- CLaMR's "1,504 query-document pairs" **is** `multivent_2_train_judgments.jsonl`
  (1,520 judgments / 1,361 queries / 417 docs) minus the 3 docs with no frame
  mapping → **exactly 1,504**. Their eval set is the *human-annotated train*
  judgments; they train on synthetic queries.
- `clamr/data/multivent_train_mapping.json` (in their repo) gives per-video
  PySceneDetect frame indices **and the shard each video lives in**, so you need
  **321 of 1,085 shards (~280 GB streamed), not 943 GB**. Download a shard, mine
  its handful of videos, delete it; peak disk stays under 10 GB.
- The shards also contain `{key}.json` (the `description` channel) and
  `{key}.m4a` (audio). Take all three in one pass — I did two passes and paid
  the bandwidth twice.
- Only 16 GB of `features/` is needed for ASR/OCR text; the CLIP/SigLIP feature
  dumps are for *other* baselines, not CLaMR, which needs raw frame pixels.

Incidental but relevant: **140 of 414 videos have fewer than 10 scene-detected
frames, and 16 have exactly one.** A third of the pool is visually thin.

## Finding 4 — the reproduction post-mortem (the expensive lesson)

First run produced **nDCG@10 = 0.2244 against a target of 0.5847**. It did not
crash. Every metric was plausible-looking.

Root cause: **the entire language model was randomly initialized.**
`transformers` 4.52 renamed Qwen2.5-VL's submodule path
(`model.layers.*` → `model.language_model.layers.*`). CLaMR's `ColQwen2_5`
subclass bypasses the rename mapping, so **not one pretrained weight loaded**.
The warning was printed — and scrolled past in a 41 KB log while grepping for
metrics.

Why it looked plausible rather than broken: identical tokens still receive
identical *random* embeddings, so MaxSim degenerates into random-projection
lexical matching. Hence OCR scored highest (0.57) and video near zero (0.027).

Two diagnostics that pinned it, both cheap and worth copying:

1. **Run with the adapter disabled.** Base-only scored 0.2251 vs 0.2244 with the
   adapter — i.e. the adapter contributed nothing, so the problem was upstream of
   it.
2. **Diff the checkpoint's key names against the model's `state_dict` keys.**
   165 keys starting `model.layers.` in the file; zero in the model.

Fixed by pinning `transformers==4.51.3` (pre-rename). After the pin, the only
`newly initialized` weights are `custom_text_proj.{weight,bias}` — the projection
head, which legitimately has none and is what the LoRA adapts.

> **Generalizable rule: always grep the model-loading warnings for "newly
> initialized" before trusting any reproduction number.** A silently
> randomly-initialized backbone yields numbers that look like a weak baseline
> rather than an error.

## Finding 5 — the reproduction, and what it took

| | reproduced here | CLaMR published |
|---|---|---|
| **nDCG@10** | **57.63** | **58.47** |
| R@1 | 25.47 | 26.7 |
| R@5 | 84.11 | 85.1 |
| R@10 | 87.37 | 88.0 |

Every metric lands ~1% low by the same margin, which suggests one small
systematic difference rather than a bug. Known deviations, any of which could
account for it: eval data rebuilt from raw shards (their Drive is 404), bf16
instead of their 4-bit quantization, and an explicit `min_pixels` (see below)
that may change frame resolution slightly.

Per-channel, which is the interesting part:

| channel | nDCG@10 |
|---|---|
| description | 65.13 |
| asr | 62.68 |
| ocr | 61.60 |
| **video** | **47.78** |

The visual utility is by far the weakest — consistent with 140 of 414 videos
having fewer than 10 scene-detected frames.

**Exact recipe that works** (as of 2026-08):

- `transformers==4.51.3`, `peft==0.15.2` — 4.52+ silently loads zero weights
- make the `qwen2_5_omni` import in `src/models/__init__.py` optional; it only
  exists in transformers >= 4.52, so the two constraints conflict as published
- add `"min_pixels": 4*28*28` next to `"max_pixels": 224*224` in **both**
  `process_combined` and `process_frames` — qwen-vl-utils >= 0.0.13 defaults the
  video minimum *above* their max, tripping `smart_resize`
- add an `image_token_id` setter to `ColQwen2_5Processor` (newer transformers
  assigns it in the parent `__init__` against their read-only property)
- drop `--quantize_4bit` unless you pin old bitsandbytes, and pass
  `--report_to none`
- `--per_device_eval_batch_size 1`: a third of videos have <10 frames and the
  batch collate stacks per-example video tensors

## Other things that cost time here

- `hf_hub_download(cache_dir=...)` keeps a blob behind the symlink, so unlinking
  the returned path frees nothing. Use `local_dir=` a temp dir and `rmtree` it,
  or 321 shards will fill your volume.
- CLaMR's model card says the backbone is `Qwen2.5-Omni-3B`; `adapter_config.json`
  says `Qwen2.5-VL-3B-Instruct`. The tensor count (506 = 36 layers × 7 modules ×
  2 + 2) confirms the config. Trust the config.
- Their eval reports `r@10_description = 1.12` — recall above 1, because the
  candidate pool is duplicated (one entry per judgment) and not normalized. Their
  R@1 of 26.7 vs R@5 of 85.1 has the same cause. Do not compare these numbers to
  differently-pooled ones.
- At least four different numbers are all called "MultiVENT 2.0 nDCG@10" in the
  literature, on different corpora and tasks (0.324 benchmark-paper baseline,
  0.586 MMMORRF full corpus, 0.585 CLaMR 1,504-query split, 0.848 MAGMaR 2026
  which is a different task). Never put two in one table.
- Missing deps surfaced one at a time (no requirements file): `bitsandbytes`,
  `torchmetrics`, `qwen-omni-utils`, `qwen-vl-utils`. Their 4-bit path also
  breaks on current bitsandbytes (`'Parameter' object has no attribute
  'compress_statistics'`); dropping `--quantize_4bit` is fine if you have the
  memory, but note it deviates from their setup.
- HF Trainer auto-enables `wandb` if installed and dies at the *logging* step
  after a full eval pass. Pass `--report_to none`.


## Finding 6 — why we stopped, and the screen that would have told us sooner

After reproducing, we measured the ceiling *before* building an operator. Three
diagnostics, all from one eval plus one dump of the per-channel score tensor:

**1. Oracle ceiling on any modality-level reduction.** Same tensor, different
operators:

| reduction | nDCG@10 |
|---|---|
| mean / sum | 51.71 |
| **max (CLaMR)** | **57.43** |
| best fixed weights | 56.76 |
| ORACLE per-query channel router | 58.99 |
| ORACLE per-query weight combiner | 59.12 |

A *perfect* router gains **+1.56**; a 64-sample random-weighting combiner **+1.69**.
Note mean and sum land *below* max, as on ViDoRe.

> **The +1.69 is NOT the linear ceiling — it measured the sampler.** 64 Dirichlet
> draws badly under-sample the 4-simplex. Solving the LP exactly (does any
> `w >= 0` rank the gold above all ~1490 distractors?) gives **89.3% of golds
> rankable first**, against max's 24.1% R@1. Decoy null at the same cardinality:
> **1.0%**, an excess of +88.3, so it is not a property of fitting 4 free
> parameters after choosing what to promote. Independently replicated by a second
> line: 11.3% flanked, 90.0% vs 0.8% decoy.
>
> Two traps inside this diagnostic. Duplicate-of-gold pool rows (median 3/query)
> have identical channel scores, so `s_gold − s_d` is exactly zero and the strict
> inequality is unsatisfiable by construction — uncorrected it reads 99.7%
> flanked with median margin 0.00000. And reproducing the reduction table does
> **not** catch this, because duplicates only become fatal inside the LP.

**The non-linear escape does not work here either.** `min` is the obvious
reduction the LP does not bound, and on this tensor it scores **32.74** nDCG@10
(z-scored 33.76) against max's 57.43. Min-rescued 8 queries versus min-vulnerable
503; on the flanked subset itself z-min reaches top-10 for 4.1% against max's
6.5%. So flanked golds are mostly unreachable by min too.

Flanking alone is therefore **not** evidence of conjunctive evidence — the
load-bearing statistic is *rescued minus vulnerable*, and here it is
overwhelmingly negative. The redundancy conclusion stands, now for a stated
reason rather than a mis-measured ceiling.

What survives untouched, because none of it is a searched ceiling: achieved max
57.43 and mean 51.71, the marginal-contribution result (deleting all frames costs
0.32), and the anti-complementarity finding (real channels 12.7 points below
noise-matched copies). Those are realised behaviour.

**And now closed.** The information to rank the gold first *is* present in the
four channel scores for ~89% of queries — but a learnable function of the query
does not find it. Fitting a weighting on the query embeddings (linear head,
listwise CE, video-disjoint 5-fold CV):

| | nDCG@10 |
|---|---|
| max | **57.43** |
| learned fixed `w` | 56.50 |
| learned `w(q)` | 56.23 |

Both learned reductions land *below* max, and conditioning on the query buys
nothing over a fixed weight. So the 89.3% LP reachability is **oracle geometry**
— `w` fitted after seeing which document is gold — not query-decodable signal.
Same shape as the VisDial null in [03](03-mmeb-visdial.md), where a gate that was
active still bought +0.2.

Caveats, stated because they bound the claim: single seed, linear probe,
post-norm features. A non-linear head or pre-norm features could differ, and the
pre-norm rerun is queued. Read it as "null at probe level", not "impossible".

The lesson generalises past this benchmark: **an oracle ceiling and a learnable
ceiling are different quantities, and the gap between them can be the whole
result.** 89.3% reachable, 0% recovered.

Striking detail: the oracle would pick video for 783 of 1504 queries where max
picks it for 38, agreeing only 35.5% of the time — completely different choices,
worth 1.6 points. That is what redundancy looks like.

**2. Unique coverage.** For each channel, how many queries does it retrieve
(gold in top-10) that *no other channel* does:

| channel | covers | unique |
|---|---|---|
| description | 85.8% | 6.8% |
| asr | 73.3% | 1.9% |
| video | 59.9% | **0.2%** (3 queries) |
| ocr | 48.1% | **0.1%** (2 queries) |
| any channel | 89.3% | — |

580 of 1504 queries are covered by all four. Rank correlations are low
(video↔asr 0.14), so the channels *rank* differently but *cover* the same
queries — a better operator can reorder, it cannot find anything new.

**3. Marginal contribution of a modality — delete it and re-run.**

| frames per video | nDCG@10 | video-query nDCG@10 |
|---|---|---|
| 10 | 57.63 | 47.78 |
| 5 | 57.40 | 47.50 |
| 2 | 57.50 | 47.56 |
| **0** | **57.31** | **47.28** |

**Deleting every frame costs 0.32 points.** 274 tokens per document, 30% of the
sequence, a full vision transformer — a third of a point.

This does *not* contradict the paper, and reading both together is the real
result. Their Table 2 shows a vision-only model reaches 40.71, so frames plainly
carry retrievable signal. Our deletion shows its marginal contribution given text
is 0.32. **Strong alone, redundant in context.** Neither number alone tells you
that; you need both.

Consequence: a query-conditioned *frame selector* is pointless here — you cannot
select your way out of a channel worth 0.32 points. And the 161 queries no
channel retrieves are **90% video-targeted** while having *longer*-than-average
descriptions, so they are a visual capability gap that neither better selection
nor more metadata addresses.

### The screen, for next time

Before building a fusion operator on any benchmark, spend one eval on:

1. **marginal contribution** — delete each modality, measure the drop
2. **unique coverage** — queries this channel finds that nothing else does
3. **oracle router / combiner** — the ceiling for any per-query linear
   reduction; per-item non-linear operators need direct evaluation (gotchas.md)

If deleting a modality is nearly free, the modalities are redundant and no
operator will pay, however well designed. Nobody publishes these numbers, which
is why this failure mode keeps costing people weeks. Scripts:
`reduce_sweep.py`, `complementarity.py` (see the repo issues for copies).

### A correction worth recording

We first compared our separate-encoding run (49.01) against the paper's 44.53 as
if the paper were off. It is not. Their variant B is **retrained** without
contextualization (Table 2: "impact of architectural and objective choices");
ours was the combined-trained checkpoint **evaluated** without it — a train/test
mismatch. Different quantities: theirs measures how much worse a model trained
without contextualization is; ours measures how much the trained model depends on
joint encoding at inference. It is coherent that our mismatched 49.01 sits above
their retrained 44.53.
