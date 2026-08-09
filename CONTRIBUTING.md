# Contributing

PRs welcome, including — especially — negative results.

## What belongs here

- An attempt at one of the open questions below.
- A correction. If a number here is wrong, say so and show why.
- Your own dead end on a related question, in the same format.
- A gotcha that cost you hours and produced a *plausible wrong number* rather
  than a crash. Those go in [gotchas.md](gotchas.md).

## Format

Keep it under ~150 lines. A future agent should be able to decide in two minutes
whether to skip a direction. Include:

1. **Verdict in the first two lines.** Worked / did not work / unresolved.
2. **The number, the baseline, and the seed count.** "Beat the baseline" without
   a seed count is not a result.
3. **What was not controlled.**
4. **The reusable conclusion** — what a reader should do differently, stated so
   it survives outside your specific setup.

Do not polish away the failures. The failures are the content.

## Open questions

- Does an explicit factor graph over separately-encoded modalities recover the
  +13.94 nDCG@10 that CLaMR gets from pushing all channels through 28 LM layers
  together? (See [04](04-multivent-clamr.md).) If it does, cross-modal fusion is
  *structure*, not depth.
- MultiVENT's shards ship real `.m4a` audio that every published method on it
  ignores. What does adding it buy?
- Is there a retrieval benchmark with signed/contradictory evidence *and*
  multiple modalities? [02](02-nevir-negation.md) is signed but text-only;
  [04](04-multivent-clamr.md) is multimodal but the evidence is co-supporting.
  The intersection is where a learned reduction should win biggest, and I have
  not found a benchmark there.
- How many published reranking gains are inside the ~1-point seed noise band
  measured in [03](03-mmeb-visdial.md)?

## Tone

Be fair to the authors of the work discussed here. Several notes point out that
a released artifact does not reproduce; state what you observed, what you
deviated on, and what remains unconfirmed. "This did not reproduce in my hands,
here is exactly what I ran" — not "this is broken".
