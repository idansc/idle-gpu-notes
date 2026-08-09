# Idle-cluster research notes

Short, honest write-ups of research directions I tried on my lab's GPU cluster
when it would otherwise have sat idle. Mostly retrieval and multimodal fusion.

Many of these are **negative results**. That is the point. A null that took two
days of GPU time and 300 GB of downloads to establish is worth writing down, and
almost nobody writes them down.

Anyone can read this, and PRs are welcome — corrections, extra datapoints, or
your own attempt at the same question. If you burned a week finding out that
something does not work, that belongs here too.

## Why this exists

Most of this work was driven by AI coding agents. Agents re-derive the same dead
ends over and over because dead ends are not published. Each file below is
written to be readable by a future agent (or human) in under two minutes, so it
can skip what has already been ruled out. The unit of value here is **tokens and
GPU-hours saved**, not novelty.

## The through-line

Nearly everything here tests one question:

> **Does a learned reduction operator beat a hand-designed one (max, mean, dot)?**

Short answer so far: **it depends entirely on whether the evidence is redundant
or signed.**

| Setting | Evidence type | Learned operator vs hand-designed |
|---|---|---|
| [ViDoRe / ColPali page retrieval](01-vidore-maxsim.md) | redundant | loses badly (81.7 vs 84.96) |
| [NevIR negation](02-nevir-negation.md) | signed | wins (21.7 → 25.4 frozen, 46.2 trained) |
| [MMEB VisDial image retrieval](03-mmeb-visdial.md) | redundant | ties exactly (+0.2 pts, gate active) |
| [MultiVENT 2.0 / CLaMR](04-multivent-clamr.md) | redundant across channels | operator bounded at +1.7 |
| [FLARE audiovisual](05-flare-audiovisual.md) | **non-redundant by construction** | in progress |
| [LongVideoBench frame selection](06-longvideo-frame-selection.md) | localized ↔ distributed *(does the best operator depend on the item?)* | ties on average; the *hand-designed winner inverts* with evidence count, so conditioning — not capacity — is the open win |

If your evidence is redundant across positions — many patches of a page all
support the same query — `max` is already near-optimal and no amount of learned
pooling capacity will help. If your evidence is *signed*, so that some matches
should **lower** the score, `max` is provably wrong, because it is monotone.

## Files

- **[01-vidore-maxsim.md](01-vidore-maxsim.md)** — learned reductions of the
  query×patch similarity grid vs MaxSim. Redundancy quantified.
- **[02-nevir-negation.md](02-nevir-negation.md)** — negation retrieval, where
  monotonicity is a provable defect. The one clear win.
- **[03-mmeb-visdial.md](03-mmeb-visdial.md)** — listwise reranking and
  query-conditioned aggregation on MMEB. Clean null, plus a warning about
  single-seed results.
- **[04-multivent-clamr.md](04-multivent-clamr.md)** — omni retrieval where one
  item carries several modalities. Full reproduction post-mortem, plus the screen
  that bounds a fusion operator before you build it.
- **[05-flare-audiovisual.md](05-flare-audiovisual.md)** — the first benchmark
  whose queries *require* two modalities, and where the incumbent fusion is a
  placeholder average that costs some models 7×.
- **[gotchas.md](gotchas.md)** — environment and library traps that cost hours
  and produce *plausible* wrong numbers rather than crashes. Read this one first
  if you are debugging a reproduction that "runs fine" but misses.

## House rules for these notes

1. State the number, the baseline, and the seed count. A single seed is not a
   result — see [03](03-mmeb-visdial.md) for a 2-point "win" that evaporated at
   n=16.
2. Say what was *not* controlled.
3. Nesting checks: if your fancy operator has a setting where it reduces to the
   baseline, verify numerically that it does. This caught two silent bugs.
4. Never claim a reproduction without matching a published number first.

## License

MIT for the notes and any code. The datasets and models referenced have their
own licenses.
