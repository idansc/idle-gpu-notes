# 05 — FLARE: the first benchmark where the modalities are actually non-redundant

**Status: in progress. Data prepared, screen not yet run. Recorded now because the
benchmark *selection* reasoning is the transferable part.**

## Why this one, after three failures

Three benchmarks failed the fusion question for three different reasons:

| benchmark | why it failed |
|---|---|
| [ViDoRe](01-vidore-maxsim.md) | one modality; evidence redundant across patches |
| [MMEB / VisDial](03-mmeb-visdial.md) | items are single-modality |
| [MultiVENT](04-multivent-clamr.md) | items multimodal, but channels mutually redundant (deleting all frames costs 0.32) |

FLARE ([arXiv 2605.10228](https://arxiv.org/abs/2605.10228), data
`YqjMartin/FLARE` CC BY 4.0 ungated, code MIT) is the first with the property the
question needs: **queries carry a hard bimodal constraint by construction.**

A real unified query from `clip-query-unified.jsonl`:

> *"hands making a leather wallet with a watch visible on the wrist, **and a sad
> violin melody in the audio**"*

Visual constraint AND acoustic constraint in one query. Neither modality alone
satisfies it. Contrast MMEB's OmniSET, which renders one meaning four ways in
parallel and never combines them.

Scale: 399 long videos (225.4 h), 87,697 clips, 274,933 queries — 53,580 unified,
86,350 vision-only, 135,003 audio-only. The single-modality query sets are
built-in controls.

## The incumbent operator is provably wrong here

This is what makes it interesting. In `retrieval_adapters.py`, the audiovisual
fusion is, verbatim:

```python
video_feat = normalize(embeddings[ModalityType.VISION])
audio_feat = normalize(embeddings[ModalityType.AUDIO])
feat = normalize((video_feat + audio_feat) / 2.0)
```

An arithmetic mean of two unit vectors. The paper is explicit that this is a
placeholder, not a design: ImageBind and LanguageBind *"provide no official recipe
for combining per-modality embeddings"*, so they use *"the standard late-fusion
practice by average pooling"*.

Reported Text→Clip R@1 (caption-based):

| model | vision-only | naive AV average |
|---|---|---|
| ImageBind | 19.04 | **7.64** |
| LanguageBind | 19.94 | **2.70** |
| Perception AV | 24.55 | 26.48 |
| WAVE-7B | 27.84 | **65.51** |

The authors diagnose it: *"the gulf between vision-only and audio-only embedding
quality [...] is so large that simple pooling is dominated by the weaker audio
component rather than being complemented by it."*

So the same fusion costs ImageBind 2.5×, costs LanguageBind 7.4×, and gains
WAVE 2.4×. That spread is the operator's doing — the opposite of ViDoRe and
MultiVENT, where the hand-designed reduction was near-optimal and unbeatable.

**Caveat we have not yet cleared:** those vision-only and unified numbers may be
computed on *different query sets* (vision queries vs unified queries). If so the
2.5× is partly a population artifact, not a fusion failure. We made exactly this
mistake once already (see the correction in [04](04-multivent-clamr.md)) and will
not inherit the pairing — the harness exposes `media_mode`, so the first run holds
the unified query set fixed and varies only the media scope.

## The ladder to test

1. naive average — the incumbent
2. **best fixed per-modality weight** — a 2-parameter fix. If this recovers most
   of the gap, the "fusion is broken" story is really "nobody tuned a scalar",
   and that is the honest baseline any learned operator must beat.
3. query-conditioned weighting — the only thing that can use audio on *"a sad
   violin melody"* and ignore it on *"text on black screen saying subscribe"*

Step 2 is the one the field skipped by inheriting a placeholder, and skipping it
would let us claim credit for a scalar.

## Practical notes

- 79 GB of videos as 14 zips of ~6,600 clip mp4s, `<video_id>/<video_id>-Scene-NNN.mp4`,
  matching the metadata `video_path` exactly. Metadata is a separate 1 GB.
- **Audio does not ship as audio.** It is inside the mp4s; every audio encoder in
  the harness wants wav paths, so 87,697 ffmpeg extractions are unavoidable.
- **No train split.** The paper evaluates pretrained retrievers zero-shot, so any
  trained result is on a split you invent — and it must be split by the 399 source
  videos, never by clip, or clips of the same video leak across the boundary.
- Harness natively supports three modality scopes (vision / audio / unified) and
  two query regimes (caption-based / query-based), so the screen below is free.

## The screen, before building anything

The three diagnostics that would have saved us on MultiVENT, plus a fourth for
the conjunctive structure this benchmark claims, run first this time:

1. **marginal contribution** — delete a modality, measure the drop
2. **unique coverage** — queries this channel retrieves that nothing else does
3. **oracle ceiling** — per query, the best channel or weighting; the bound on
   per-query *linear* operators only (gotchas.md)
4. **Pareto-flanking count** — golds no linear weighting can rank first (LP
   feasibility, one constraint per distractor) that `min` still ranks first: the
   operator-free measure of score-level conjunctiveness. Pre-registered
   prediction: calibrated-min (per-channel affine, then min) beats the best
   fixed weighted mean iff this count is material. Calibrate the diagnostic on
   MultiVENT's tensor first, where it should be ~0.
   Binds on the calibrated-min arm, so the comparison is one-variable: both
   arms score the SAME per-channel affine-calibrated inputs (weighted mean of
   calibrated vs min of calibrated); the calibration is fit on a video-level
   holdout (split by the 399 source videos, never by clip); and the gap is
   pre-registered to appear on unified queries only, with vision-only queries
   as the ~0-gap control

If audio's marginal contribution on unified queries is large, this benchmark can
support the fusion question and the operator work is justified. If it is ~0.3 like
MultiVENT's frames, we stop and write up four nulls instead of three.
