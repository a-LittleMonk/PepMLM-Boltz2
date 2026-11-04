# PepMLM &rarr; Boltz-2 peptide binder pipeline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/a-LittleMonk/PepMLM-Boltz2/blob/main/PepMLM_Boltz2_colab.ipynb)

A Colab notebook that designs **linear peptide binders** against a target structure you
supply, then throws away the ones that are not predicted to bind where you asked.

[PepMLM](https://github.com/programmablebio/pepmlm) generates the peptides from the
target's sequence alone. [Boltz-2](https://github.com/jwohlwend/boltz) cofolds each one
with the real receptor and supplies the numbers that decide which survive.

## Why this exists

PepMLM's repository is two files: a generator and a training script. There is no
filtering, no structure, and no ranking beyond its own pseudo-perplexity — which
correlates with AlphaFold interface confidence at only **r = &minus;0.41**, by the
authors' own measurement. Generation without a cofolding filter is close to useless,
and no published pipeline closes that gap.

This notebook is the missing half.

## The funnel

| Step | | Cost |
|---|---|---|
| 1 | Set up everything | ~20 min, once per session |
| 2 | Target: chain and crop | ~2 min, builds the receptor MSA once |
| 3 | Generate thousands of peptides, gate on perplexity | minutes |
| 4 | Cofold the survivors with Boltz-2, steered at your hotspots | **hours — the entire cost** |
| 5 | Filter on interface, hotspots and synthesisability | seconds |
| 6 | Look at a result | — |

Steps 3, 4 and 5 are separate deliberately: re-filter without re-folding, re-fold
without re-generating.

## Hotspots, in two independent places

Give the residues you want hit **in your own file's numbering**, and give them to
whichever of the two steps you mean. They are deliberately separate:

- **Step 4 steers.** Boltz-2 takes a native `pocket` constraint, so the peptide is
  folded against the site you named instead of wherever the model prefers. Leave it
  empty to let Boltz choose freely.
- **Step 5 judges.** After folding, it measures which receptor residues the peptide
  actually touches. `hotspot_contacts` counts how many of yours were hit;
  `hotspot_fraction` is the share of all its contacts landing on the patch around them.
  That second number is what removes a peptide that binds confidently **to the wrong
  face** — a failure the confidence metrics alone will happily pass.

Keeping them apart means you can fold once and then ask several different questions of
the same designs. Changing step 5's hotspots, `patch_radius` or `contact_cutoff`
re-measures from the saved structures and refolds nothing, so it costs seconds. Both
steps print the residue each number resolves to, and refuse a residue the target does
not contain before doing any work.

Step 2 holds no hotspots at all. It only defines the target: the chain, the crop, the
template and the alignment.

## Design decisions worth knowing

**The MSA is built once.** Boltz generates an MSA per input file, so 100 peptides would
mean 100 MMseqs2 server queries and a rate-limit ban. Step 2 runs the receptor alone,
keeps the alignment, and every peptide reuses it with `msa: empty` on the peptide chain.
The alignment is cached in your Drive by sequence hash, so a second run against the same
target skips it entirely. That single run also predicts the **free receptor**, which the
filter uses to report how far the target had to move.

**Your PDB becomes a template.** The receptor is rewritten as a single chain renumbered
from 1 and handed to Boltz as a template, so it keeps its real fold.

**Sampling is restricted to the 20 standard amino acids.** PepMLM samples from the whole
ESM-2 vocabulary, which includes `<mask>`, `<pad>`, `X` and the rare letters B/U/Z/O.
Those get silently dropped when the peptide is decoded, so upstream's script can return
a peptide **shorter than you asked for**. Masking the logits to the 20 standard residues
fixes that and keeps every peptide synthesisable.

**One forward pass per length, not per peptide.** PepMLM's own loop re-runs the model
for every binder, but the input is the same fully-masked sequence each time and the model
is deterministic in eval mode — the logits are identical. Sampling *n* peptides from one
pass is the same computation, hundreds of times faster. Pseudo-perplexity is still
computed exactly as published, position by position, batched across peptides.

**Perplexity is ranked within each length, never pooled.** It scales with length, so a
global top-N would select on arithmetic rather than quality.

**The 2 GB affinity checkpoint is skipped.** Boltz downloads it on every run, but it
only works on small molecules and is never loaded unless a YAML requests it. A
placeholder keeps Boltz quiet.

## Output

```
My Drive/PepMLM/<run_name>/
  master.csv            every peptide generated, every metric, unfiltered
  final.csv             the passing set, ranked
  structures/           CIF for everything cofolded
  passing/              CIF for the final set, named by rank
  settings.json         the generation parameters
  filter_settings.json  the thresholds that produced final.csv
My Drive/PepMLM/inputs/      <- put your PDB or CIF here
My Drive/PepMLM/receptors/   <- cached MSAs, keyed by sequence hash
```

`master.csv` is rewritten after every chunk of cofolding, so a Colab disconnect costs
one design. Re-running step 4 resumes.

## Reading the numbers

| Column | Rough reading |
|---|---|
| `perplexity` | lower is better; **not comparable between lengths** |
| `iptm_pair` | receptor&ndash;peptide interface confidence: >0.6 interesting, >0.75 good |
| `ipae` | interface error in &aring;ngstroms: <10 interesting, <7 good |
| `pep_plddt` | is the peptide confidently placed: >0.7 |
| `hotspot_contacts` | how many of your residues were hit |
| `hotspot_fraction` | <0.5 means the wrong face |
| `receptor_rmsd` | how far the target moved vs. its free prediction |
| `net_charge`, `gravy`, `n_cys`, `liabilities` | whether you can make it |

## A T4 is tight

Cofolding is the whole budget. Rough figures for a ~250-residue receptor at screening
settings:

| GPU | per design | 100 designs |
|---|---|---|
| T4 (free) | ~2&ndash;4 min | 4&ndash;6 h, may run out of memory on a large receptor |
| L4 | ~40&ndash;60 s | ~1.5 h |
| A100 | ~15&ndash;30 s | ~40 min |

Step 1 detects the GPU and sets its defaults from it. Below compute capability 8.0,
Boltz runs with `--no_kernels`, which is slower but numerically the same.

## What this is not

A peptide that clears every gate here has passed a **structure-prediction** check, not a
binding assay. Boltz-2 trained on the PDB, which overlaps PepMLM's training data, so
part of its confidence is circularity rather than physics — and on short linear peptides
its ipTM is a far weaker signal than on folded miniproteins.

Treat `final.csv` as a shortlist worth synthesising in which most entries will still not
bind. If you want better odds per molecule, a structure-based designer (RFdiffusion,
BindCraft, PXDesign) is the stronger start — at the cost of a miniprotein you cannot
make by solid-phase synthesis.

## Building it

The notebook is generated, not edited by hand. Each cell is a file under `cells/`:

```
python build_nb.py     # compiles every cell, writes PepMLM_Boltz2_colab.ipynb
python -I test_cells.py # 20 checks: numbering, YAML, contact geometry, filter gates
```

`test_cells.py` runs without a GPU. It builds a deliberately awkward PDB — non-sequential
numbering, a chain break, an insertion code, a water, a second chain — and checks the
crop against it, checks the hotspot mapping in both steps that do it, validates the
emitted YAML, measures contacts on
synthetic complexes placed on the right and wrong faces, and drives each filter gate with
a design built to fail exactly that gate.

## Credit

- **PepMLM** — Chen, Quinn, Dumas et al., *Target sequence-conditioned design of peptide
  binders using masked language modeling*, Nature Biotechnology 44, 1002–1010 (2025).
  [repo](https://github.com/programmablebio/pepmlm) ·
  [paper](https://www.nature.com/articles/s41587-025-02761-2)
  The live weights are `ChatterjeeLab/PepMLM-650M`; the `TianlaiChen/PepMLM-650M` link in
  their README is dead.
- **Boltz-2** — Passaro, Corso, Wohlwend et al. (2025), MIT licensed, code and weights.
  [repo](https://github.com/jwohlwend/boltz)
- MSAs come from the ColabFold MMseqs2 server, queried **once per target sequence**.

PepMLM ships no licence file — ask the authors before commercial use.
