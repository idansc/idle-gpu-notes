# Publication candidates

When a line finds something worth publishing, add it here by PR, in this format:
**what we found** (one sentence a non-specialist can read), **why it matters**
(one sentence), the numbers, the note, and citations for any claim about what
the field does. The critic verifies the controls; Idan decides the venue.
Assembly drafts live in `papers/`.

---

## Ready — numbers on main, instruments validated

### 1. "Oracle headroom" can be an illusion

**What we found:** for 9 of 10 MultiVENT queries, SOME weighting of the four
channel scores would rank the right video first (89.3%, vs 0.8% for a random
decoy) — but a model trained to produce that weighting from the query alone
gains nothing at probe level (56.23 vs plain max 57.43; single seed, linear
probe, post-norm features — pre-norm rerun queued). The headroom is real
geometry and, at least at probe level, absent information.

**Why it matters:** oracle and upper-bound gaps are standard motivation in
selection/fusion work — Frame-Voyager builds its training signal from
oracle-style ranking of frame combinations
([2410.03226](https://arxiv.org/abs/2410.03226)), and our own ledger fell for
one twice (note 04's retracted +1.69 "hard ceiling"; note 06's retracted
74.0 per-item oracle, 98% chance harvesting). We show such gaps can be
unreachable in principle from the query side, and we ship the one-eval test
(LP feasibility + decoy null + learnability probe). (Note 04.)

### 2. Frame selection is worth ~2.5 points, not the ~10 it appears

**What we found:** on 4-option MCQ, two equal-quality runs already disagree on
~8% of items by chance (churn floor 8.12); the raw flip-set reads 10.7 points
but the churn-corrected market for choosing WHICH 16 frames a frozen model sees
is 2.5 ± 1 on LongVideoBench ([2407.15754](https://arxiv.org/abs/2407.15754)).

**Why it matters:** the frame-selection literature's typical claimed deltas are
1–2 points against uniform or CLIP top-k — AKS
([2502.21271](https://arxiv.org/abs/2502.21271)), Frame-Voyager
([2410.03226](https://arxiv.org/abs/2410.03226)), Q-Frame
([2506.22139](https://arxiv.org/abs/2506.22139)), QCA
([2607.00983](https://arxiv.org/abs/2607.00983)), FOCUS
([2510.27280](https://arxiv.org/abs/2510.27280)), AGFS
([2603.20180](https://arxiv.org/abs/2603.20180)), ReQuest
([2607.01737](https://arxiv.org/abs/2607.01737)) — i.e., inside or near the
noise floor we measure, and none report a paired null of this kind. Precedent
for the audit-paper shape in another domain: "A Sober Look at Progress in
Language Model Reasoning" ([2504.07086](https://arxiv.org/abs/2504.07086)).
(Note 06.)

### 3. The field's default frame-picking rule inverts on multi-evidence questions

**What we found:** query–frame similarity top-k (the default rule, as in AKS
[2502.21271](https://arxiv.org/abs/2502.21271) and Q-Frame
[2506.22139](https://arxiv.org/abs/2506.22139)) beats uniform spacing by 8
points on questions needing ONE moment and falls BELOW uniform on questions
needing ≥3 — and benchmark averages cancel the two regimes.

**Why it matters:** leaderboard averages cannot distinguish a better method
from a luckier mix of question types; the stratifying variable (evidence count)
is annotated in LongVideoBench and computable elsewhere. AGFS
([2603.20180](https://arxiv.org/abs/2603.20180)) observed per-category
preference heterogeneity; evidence count is the per-item causal variable
behind it. (Note 06.)

---

## Pending — expected to reach this bar

### FLARE fusion screen (grid running)

FLARE ([2605.10228](https://arxiv.org/abs/2605.10228)) is the first benchmark
whose queries REQUIRE two modalities, and where the incumbent fusion — average
pooling of L2-normalized embeddings, the "standard late-fusion practice" in
FLARE's own words, over ImageBind
([2305.05665](https://arxiv.org/abs/2305.05665)) / LanguageBind
([2310.01852](https://arxiv.org/abs/2310.01852)) — costs some models 2.5–7×
rather than 2 points. First completed runs: ImageBind's audio channel is
near-dead on user-style queries (R@10 = 1.65% on an 87,697-clip gallery), with
the pipeline exonerated end-to-end (its own demo pairs and real FLARE clips
align 5/5 with wide margins), and replicated cross-hardware (mm-lab01 mirror:
R@10 1.655 vs cluster 1.652). If confirmed, the field's default fusion averages
in a dead channel, and the operator question moves to strong-audio substrates.

**Venue check (2026-08-11): ICASSP 2027, submission 2026-09-16 (~5 weeks).**
Novelty verdict: the OBSERVATION (audio weak on FLARE) is in the FLARE paper
itself ("audio-language alignment remains a key bottleneck"), and audio-text
query-robustness is an active lane — Omni-Embed-Audio
([2604.18360](https://arxiv.org/html/2604.18360)) reports CLAP "collapse
ratios" by query formulation, and Robust Audio-Text Retrieval
([2604.23323](https://arxiv.org/html/2604.23323v1)) hardens the same. What is
NOT taken and shapes the 4-pager: the dead-CHANNEL audit for audio-VISUAL
retrieval — alive-vs-aligned distinction with pipeline exoneration, the
fusion-damage mechanism (averaging in a dead channel destroys a healthy one),
cross-hardware replication, and substrate-conditional recovery on strong-audio
encoders (WAVE-7B / Qwen-Omni, staged). Position against both papers above and
against FLARE's own bottleneck sentence, or a reviewer will. Gate to commit:
vision/unified rows + one strong-audio contrast must land first (in flight,
same week).

### Imaginability audit (H3 line, design locked)

Swap a benchmark's real video for one GENERATED from the question alone
(MiniMax H3); if the score barely drops, the benchmark is not testing video
understanding. A format-matched, portable validity meter for any video
benchmark — constructive where TVBench
([2410.07752](https://arxiv.org/abs/2410.07752)) diagnosed MVBench
([2311.17005](https://arxiv.org/abs/2311.17005)) destructively (shuffle,
single-frame, blind). Adjacent trend it audits rather than joins: video models
as reasoners — Veo 3 ([2509.20328](https://arxiv.org/abs/2509.20328)), Thinking
with Video ([2511.04570](https://arxiv.org/abs/2511.04570)).

### "Search can't say no" (NegBench line, substrate gate first)

Every indexable retrieval scorer — bi-encoder cosine (CLIP,
[2103.00020](https://arxiv.org/abs/2103.00020)), late-interaction MaxSim
(ColBERT, [2004.12832](https://arxiv.org/abs/2004.12832)), sparse expansion
(SPLADE, [2107.05720](https://arxiv.org/abs/2107.05720)) — is monotone:
matching more never lowers a score, so "a photo without a frisbee" scores
HIGHER on frisbee photos. NevIR ([2305.07614](https://arxiv.org/abs/2305.07614))
and NegBench ([2501.09425](https://arxiv.org/abs/2501.09425)) measure the
failure (near-chance); the 2026 SOTA fixes are non-indexable listwise LLM
rerankers (SIGIR'25 reproduction,
[2502.13506](https://arxiv.org/abs/2502.13506)) — "saying no costs quadratic".
We formalize the impossibility and test the cheapest indexable signed fix on
frozen encoders (NevIR frozen prior: +3.7).
