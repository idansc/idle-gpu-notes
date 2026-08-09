# 04 — MultiVENT 2.0 / CLaMR: finding a benchmark with genuinely multimodal items

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

A *perfect* router gains **+1.56**; a perfect combiner **+1.69**. That is the ceiling
for any per-query routing or weighting of these four scores; per-item
non-linear reductions (min, product) sit outside that family and are not
bounded by it — see gotchas.md. Here the distinction is moot: max was swept
directly (57.43) and the channels are redundant, so the conclusion stands. Note mean and sum
land *below* max, as on ViDoRe.

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
