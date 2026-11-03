## Reading the numbers

| Column | What it is | Rough reading |
|---|---|---|
| `perplexity` | PepMLM's own confidence. **Not comparable between lengths.** | lower is better |
| `iptm_pair` | Boltz-2 interface confidence, receptor&ndash;peptide specifically | >0.6 interesting, >0.75 good |
| `ipae` | interface predicted error, &aring;ngstroms | <10 interesting, <7 good |
| `pep_plddt` | is the peptide itself confidently placed | >0.7 |
| `hotspot_contacts` | how many of your residues it actually touches | the more the better |
| `hotspot_fraction` | share of its contacts on the patch around your hotspots | <0.5 means it is on the wrong face |
| `receptor_rmsd` | how far the receptor moved vs. the free prediction | large = a pose that needs the target to deform |
| `net_charge`, `gravy`, `n_cys`, `liabilities` | can you actually make it | see step 5 |

## What this is not

A peptide that passes every gate here has passed a **structure-prediction** check, not
a binding assay. Boltz-2 was trained on the PDB, which overlaps the data PepMLM was
trained on, so some of its confidence is circularity rather than physics. On short
linear peptides its ipTM is a much weaker signal than it is on folded miniproteins.

Treat `final.csv` as a shortlist worth synthesising, in which most entries will still
not bind. If you need higher prior odds per peptide, a structure-based designer
(RFdiffusion, BindCraft, PXDesign) that builds a folded miniprotein is the better
starting point — at the cost of a molecule you cannot make by solid-phase synthesis.

## Re-folding the finalists properly

Step 4 runs at `sampling_steps = 50` to screen cheaply. Once `final.csv` is short,
raise it to Boltz's default **200**, set `diffusion_samples` to 3&ndash;5, tick
`redo`, and run step 4 again on just the finalists (set `max_designs` to their count
after re-running step 3 with a tighter gate). Metrics that survive that are worth
more than metrics that only existed at 50 steps.

## Credit

- **PepMLM** &mdash; Chen, Quinn, Dumas et al., *Target sequence-conditioned design of
  peptide binders using masked language modeling*, Nature Biotechnology (2025).
  [repo](https://github.com/programmablebio/pepmlm) &middot;
  [paper](https://www.nature.com/articles/s41587-025-02761-2)
- **Boltz-2** &mdash; Passaro, Corso, Wohlwend et al. (2025).
  [repo](https://github.com/jwohlwend/boltz)
- MSAs come from the ColabFold MMseqs2 server. Please do not hammer it: this notebook
  queries it **once per target sequence** and caches the result in your Drive.
