# PepMLM &rarr; Boltz-2 peptide binder pipeline

Generate linear peptide binders against your own target with
[PepMLM](https://github.com/programmablebio/pepmlm), cofold every candidate with
[Boltz-2](https://github.com/jwohlwend/boltz), and keep only the ones predicted to
bind **where you asked**.

PepMLM on its own gives you sequences and a self-assigned score that correlates with
AlphaFold confidence at only **r = -0.41** — it is an idea generator, not a filter.
This notebook supplies the filter.

**The funnel**

| | | cost |
|---|---|---|
| 1 | Set up everything | ~20 min, once per session |
| 2 | Target: chain, crop, hotspots | ~2 min (builds the receptor MSA once) |
| 3 | Generate a few thousand peptides and gate on perplexity | minutes |
| 4 | Cofold the survivors with Boltz-2 | **hours — this is the whole cost** |
| 5 | Filter on interface quality, hotspots and synthesisability | seconds |
| 6 | Look at a result | — |

Steps 3, 4 and 5 are separate on purpose: you can re-filter without re-folding, and
re-fold without re-generating.

**Two files land in your Drive:** `master.csv` is every peptide generated with all of
its cofolding metrics, unfiltered. `final.csv` is the passing set, ranked.

Licences: PepMLM has no licence file in its repo — ask the authors before commercial
use. Boltz-2 is MIT, code and weights, commercial use included.
