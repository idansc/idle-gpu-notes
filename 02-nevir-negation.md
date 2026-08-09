# 02 — Negation retrieval: where `max` is provably wrong

**Verdict: the one clear win. Learned reduction beats max when evidence is signed.**

## Why this setting

[NevIR](https://arxiv.org/abs/2305.07614) pairs two nearly identical documents
that differ only by a negation, and asks a retriever to tell them apart. This is
the complement of [01](01-vidore-maxsim.md): the evidence is not redundant, it is
**signed**.

The argument for why `max` must fail here is not empirical, it is structural:

> `max` is **monotone**. Matching a term can only ever *raise* the score.
> But under negation, matching the negated term should *lower* it.

A hand-designed monotone reduction cannot express "this match is evidence
against". That is a capability gap, not a tuning gap — which is exactly the
setting where a learned operator should pay.

## Results

| Setup | Pairwise accuracy |
|---|---|
| MaxSim baseline | 21.7 |
| learned reduction, **frozen** encoder | 25.4 |
| learned reduction + encoder training | **46.2 ± 0.1** |

The frozen-encoder number is the interesting one: +3.7 with the encoder
untouched isolates the operator, since nothing else could have changed.

## The caveat that matters

Once the encoder is trained, **the head's contribution gets absorbed**. Most of
the 46.2 is the encoder learning negation-sensitive representations, not the
reduction operator doing the work. If you only report the trained number you
will overstate the operator's contribution considerably.

This keeps recurring (see [03](03-mmeb-visdial.md)): **a trainable encoder
downstream will learn around your operator.** To measure an operator, freeze the
encoder, or run a matched arm where only the operator differs.

## Reusable conclusion

Learned reductions earn their keep when the task needs **non-monotone** scoring:
negation, contradiction, exclusion, constraint violation. Look for tasks where
some match should *reduce* relevance. If every match is positive evidence, go
back to [01](01-vidore-maxsim.md).
