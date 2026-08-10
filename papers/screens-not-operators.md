# Screens, not operators — assembly skeleton

**Thesis: whether a learned fusion/pooling/selection operator can pay on a given
benchmark is decidable-when-no in one eval, before any training; the field's
recurring learned-operator 2×2 is usually decided before it is run.** A failed
screen kills the direction; a passed screen only licenses the attempt
(flanked-but-unrescuable, note 04, is the standing counterexample).

Status: skeleton for assembly. Every number below is on main; open slots marked.

## Contributions

1. The dichotomy with mechanism: redundant evidence → hand-designed reductions
   are near-sufficient statistics; signed/conjunctive evidence → capability gap
   a learned/non-monotone operator can fill.
2. The decision procedure: marginal contribution → unique coverage →
   flanked/vulnerable net (with decoy null) → license or kill.
3. The audit: five controlled nulls + two controlled positives, one lab, paired
   arms throughout.
4. Instrument-validity discipline as a first-class method: null oracles, churn
   floors, decoy nulls, degrees-of-freedom counting — each retired at least one
   of our own claims before it retired anyone else's.

## Evidence table (claims → numbers → notes)

| Setting | Attempt | Result | The screen that predicted it | Note |
|---|---|---|---|---|
| ViDoRe page retrieval | learned grid reduction | 81.7 vs MaxSim 84.96 | argmax-deletion costs 0.7 recall | 01 |
| MMEB VisDial rerank | query-conditioned aggregation | +0.2, gate active | single-utility item = redundant regime | 03 |
| MultiVENT channel fusion | min / calibrated-min | 32.74 vs max 57.43; net 8 rescued / 503 tanked | flanked 11.3% but unrescuable (decoy null 0.8%) | 04 |
| MultiVENT learnable weighting | w(q), video-disjoint CV | 56.23 vs max 57.43, LP says 89.3% reachable | oracle geometry is not query-decodable information | 04 |
| LVB frame selection | any reshuffle at fixed budget | market 2.5 ± 1 | churn floor 8.12 on 4-way MCQ | 06 |
| NevIR negation (POSITIVE) | signed reduction, frozen encoder | +3.7 | monotone max provably wrong on signed evidence | 02 |
| CLaMR contextualization (POSITIVE) | joint vs separate encoding | self-produced gap 8.62 (paper's retrained ablation: +13.94) | cross-modal structure, not score-level reduction | 04 |

## Figures

- F1 dichotomy: redundant vs signed/conjunctive evidence, where each result sits
- F2 oracle-vs-learnable: 89.3% LP-reachable, 0.8% decoy, 0% recovered by w(q)
- F3 the screen as flowchart, kill/license semantics explicit
- F4 instrument validity before/after: subset oracle 74.0→~2 at R=1; flip-set 10.7→2.5 over churn floor

## Section plan

1. Introduction — the 2×2 everyone runs, and why it keeps tying
2. The screens (one eval each; scripts released)
3. Five audits (table above, one subsection each)
4. When operators pay: signed evidence and representation-level structure
5. Instrument validity: oracles and floors that lie in both directions (rule 8)
6. Related: Sober Look (2504.07086) for seed-noise audits; TVBench (2410.07752)
   as destructive-control precedent; ours are constructive and portable
7. Release: reduce_sweep, complementarity, flanking-count, churn-floor scripts

## Open slots

- FLARE validity example ("licenses yes" or fourth null) — grid running,
  jobs 21095897-903; three diagnostics + min-family rows land here
- Imaginability screen (H3 line) as the strict generalization of rule 6 —
  numbers pending; becomes rule 9 and a subsection of §2 when they land
