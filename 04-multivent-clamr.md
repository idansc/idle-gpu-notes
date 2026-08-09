# 04 — MultiVENT 2.0 / CLaMR: finding a benchmark with genuinely multimodal items

**Status: in progress. Reproduction post-mortem below is already useful.**

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

Fix in progress: pin `transformers==4.51.3` (pre-rename).

> **Generalizable rule: always grep the model-loading warnings for "newly
> initialized" before trusting any reproduction number.** A silently
> randomly-initialized backbone yields numbers that look like a weak baseline
> rather than an error.

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
