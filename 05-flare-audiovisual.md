# 05 — FLARE: the first benchmark where the modalities are actually non-redundant

**Status: screen complete. The hand-designed operator destroys information here —
the first such case in five benchmarks.**

## The result

Controlled screen: the same 53,580 unified queries throughout, only `media_mode`
varies. ImageBind, gallery 87,697 clips / 399 videos, zero missing records.

| query set | media | R@1 | R@5 | R@10 | MedR | n |
|---|---|---|---|---|---|---|
| unified | **vision** | **12.61** | 29.22 | 38.56 | 23 | 53,580 |
| unified | audio | 0.25 | 0.99 | 1.65 | 3356 | 53,580 |
| unified | **both (avg)** | **5.86** | 15.12 | 21.24 | 95 | 53,580 |
| vision | vision | 31.00 | 54.85 | 64.39 | 4 | 86,350 |
| vision | both (avg) | 10.32 | 22.50 | 29.68 | 51 | 86,350 |
| audio | audio | 0.12 | 0.47 | 0.78 | 8718 | 135,003 |
| audio | both (avg) | 0.97 | 2.74 | 4.08 | 2790 | 135,003 |

**On queries built to require both modalities, `normalize((v+a)/2)` scores 5.86
against 12.61 for vision alone — 2.15× worse.** The vision-query control
collapses the same way, 31.00 → 10.32.

Mechanism, exactly as the paper describes but now measured on a fixed query set:
audio alone is 0.25 (chance = 1/87,697 = 0.001%, so ~220× chance — live but very
weak), and averaging that into a 12.61 vision signal halves it. The reduction is
dominated by the weaker component.

Audio-liveness gate passes, so the weakness is the audio-text *embedding*, not
our extraction: 240 wavs across 60 videos, median RMS 0.067, 0.8% silent,
cross-clip spectral cosine −0.004 (p95 0.308).

**Calibration.** The paper's query-based Table 3 has no ImageBind vision-only row
— the single ImageBind row is the *fused* variant at 6.35/16.59/23.09. Our fused
arm on unified queries is 5.86, i.e. −7.7%: acceptable harness agreement. Our
vision-only 31.00 has no published counterpart but brackets sanely inside the
paper's own vision group (CLIP 13.89 < 31.00 < SigLIP2 33.98), consistent with
ImageBind's ViT-H tower.

So the **vision-only vs fused contrast on query-based queries is not in the
paper** — it is new content, and it is what shows the operator is the problem
rather than the encoder.

**Caveat that shaped the design:** the three query sets differ in difficulty
(vision-only scores 31.00 on vision queries but 12.61 on unified queries; audio
scores *higher* on unified queries than on audio queries). Compare only within a
fixed query set. This is why the screen holds the query set fixed instead of
inheriting the paper's cross-table pairing — and why the earlier "19.04 → 7.64"
caveat mattered.

---

*(Original note below; the benchmark-selection reasoning is the transferable
part.)*

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
   operator-free measure of score-level conjunctiveness. The diagnostic
   reports TWO counts, because a material flanked count is *necessary* for
   calibrated-min (per-channel affine, then min) to beat the best fixed
   weighted mean, not sufficient: min's failure mode is brittleness, tanking
   golds whose weakest calibrated channel sits within calibration noise of the
   distractor field. Pre-registered prediction: calibrated-min wins iff
   flanked-gold gains minus min-vulnerable losses clear zero. Calibration run
   on MultiVENT's tensor (done, CPU): flanked golds are 11.3% (170/1504) — the
   "~0 expected" guess was wrong — but the instrument is valid (decoy null:
   0.8% of random non-golds are linearly rankable-first vs 90% of golds), and
   the NET is what note 04 predicts: min-rescued 8 vs min-vulnerable 503
   (z-min nDCG@10 = 33.8 vs max 57.4; even on the flanked subset z-min puts
   4.1% in top-10 vs max's 6.5%). Flanked-but-unrescuable means flanking alone
   is not conjunctiveness — on FLARE, read only the net.
   Binds on the calibrated-min arm, so the comparison is one-variable: both
   arms score the SAME per-channel affine-calibrated inputs (weighted mean of
   calibrated vs min of calibrated); the calibration is fit on a video-level
   holdout (split by the 399 source videos, never by clip); and the gap is
   pre-registered to appear on unified queries only, with vision-only queries
   as the ~0-gap control

If audio's marginal contribution on unified queries is large, this benchmark can
support the fusion question and the operator work is justified. If it is ~0.3 like
MultiVENT's frames, we stop and write up four nulls instead of three.
