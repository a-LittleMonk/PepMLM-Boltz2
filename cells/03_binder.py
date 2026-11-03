#@title 3 &middot; Binder settings, and generate with PepMLM { display-mode: "form" }
#@markdown PepMLM makes one peptide of a fixed length at a time, so this sweeps the
#@markdown range. Generation is cheap; cofolding in step 4 is what costs time.
length_min = 10  #@param {type:"slider", min:5, max:40, step:1}
length_max = 25  #@param {type:"slider", min:5, max:40, step:1}
n_per_length = 100  #@param {type:"integer"}
#@markdown `top_k` is how many amino acids each position may be sampled from. 1 is
#@markdown deterministic, 3 is the published default, higher is more diverse:
top_k = 3  #@param {type:"slider", min:1, max:10, step:1}
#@markdown How to pick which peptides go on to cofolding. Perplexity is **not
#@markdown comparable between lengths**, so the top-N gate ranks within each length:
gate = "top N per length"  #@param ["top N per length", "perplexity cutoff", "everything"]
keep_top_n = 6  #@param {type:"integer"}
ppl_cutoff = 4.0  #@param {type:"number"}
precision = "float16"  #@param ["float16", "float32"]
seed = 0  #@param {type:"integer"}

import json, os, time
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

E = json.load(open("/content/pepmlm/env.json"))
T = json.load(open("/content/pepmlm/target.json"))
WORK, OUT, SEQ = E["work"], T["out"], T["seq"]

assert length_min <= length_max, "length_min is above length_max"
assert length_max <= T["max_peptide"], (
    f"this receptor is {len(SEQ)} residues, so the longest peptide ESM-2 can "
    f"handle is {T['max_peptide']}")

# ------------------------------------------------------------------ model
DT = torch.float16 if precision == "float16" else torch.float32
# Reload rather than upcast: a model already cast down to fp16 cannot be made
# fp32 again without having lost precision.
if "_PEP" in globals() and _PEP[1].dtype != DT and DT == torch.float32:
    del _PEP
if "_PEP" not in globals():
    print("loading PepMLM-650M ...", flush=True)
    _tok = AutoTokenizer.from_pretrained(E["repo"], cache_dir=f"{E['hf']}/hub")
    _mdl = AutoModelForMaskedLM.from_pretrained(E["repo"], cache_dir=f"{E['hf']}/hub")
    _PEP = (_tok, _mdl.to("cuda").eval())
tok, mdl = _PEP
mdl = mdl.to(DT)

# PepMLM samples from the whole vocabulary, which also holds <mask>, <pad>, X and the
# rare letters B/U/Z/O. Those are then dropped when the peptide is decoded, which can
# silently return a peptide shorter than asked for. Restricting the choice to the 20
# standard amino acids keeps every peptide exactly `length` long and synthesisable.
AA = "ACDEFGHIKLMNPQRSTVWY"
_ids = [tok.convert_tokens_to_ids(a) for a in AA]
assert all(i is not None and i >= 0 and i != tok.unk_token_id for i in _ids), \
    f"the tokenizer does not know every amino acid: {dict(zip(AA, _ids))}"
AA_IDS = torch.tensor(_ids, device="cuda")
K = min(int(top_k), len(AA))

TOKBUDGET = 20000 if E["gpu_mem"] > 20000 else 7000


@torch.no_grad()
def sample_length(L, n):
    """One forward pass, then n draws from it.

    PepMLM's own loop re-runs the model for every binder, but the input is the same
    fully-masked sequence each time and the model is deterministic in eval mode, so
    the logits are identical. Sampling n times from one pass is the same computation.
    """
    ids = tok(SEQ + "<mask>" * L, return_tensors="pt").to("cuda")
    logits = mdl(**ids).logits[0]
    pos = (ids["input_ids"][0] == tok.mask_token_id).nonzero(as_tuple=True)[0]
    lg = logits[pos].float()[:, AA_IDS]                 # (L, 20)
    topv, topi = lg.topk(K, dim=-1)
    probs = torch.softmax(topv, dim=-1)
    draw = torch.multinomial(probs, n, replacement=True)  # (L, n)
    chosen = topi.gather(-1, draw)                        # (L, n)
    out = []
    for j in range(n):
        out.append("".join(AA[i] for i in chosen[:, j].tolist()))
    return out


@torch.no_grad()
def perplexities(peptides):
    """Pseudo-perplexity: mask each peptide position in turn, batched across peptides."""
    rows, owner, truth = [], [], []
    for pi, pep in enumerate(peptides):
        ids = tok(SEQ + pep, return_tensors="pt")["input_ids"][0]
        n = len(pep)
        for k in range(n):
            j = len(ids) - 1 - n + k          # last token is <eos>
            r = ids.clone()
            truth.append(int(r[j]))
            r[j] = tok.mask_token_id
            rows.append((r, j))
            owner.append(pi)
    width = len(rows[0][0])
    per_batch = max(1, TOKBUDGET // width)
    loss_sum = np.zeros(len(peptides))
    loss_cnt = np.zeros(len(peptides))
    for s in range(0, len(rows), per_batch):
        sl = rows[s:s + per_batch]
        batch = torch.stack([r for r, _ in sl]).to("cuda")
        at = torch.tensor([j for _, j in sl], device="cuda")
        lg = mdl(input_ids=batch).logits.float()
        lp = torch.log_softmax(lg[torch.arange(len(sl), device="cuda"), at], dim=-1)
        tr = torch.tensor(truth[s:s + per_batch], device="cuda")
        nll = -lp.gather(-1, tr[:, None])[:, 0].cpu().numpy()
        for o, v in zip(owner[s:s + per_batch], nll):
            loss_sum[o] += v
            loss_cnt[o] += 1
    return np.exp(loss_sum / np.maximum(loss_cnt, 1))


# ------------------------------------------------------------------ generate
torch.manual_seed(int(seed))
lengths = list(range(int(length_min), int(length_max) + 1))
print(f"\n{len(lengths)} lengths x {n_per_length} samples "
      f"= {len(lengths) * n_per_length:,} peptides against {len(SEQ)} residues\n")

rows, t0 = [], time.time()
for L in lengths:
    peps = sorted(set(sample_length(L, int(n_per_length))))
    ppl = perplexities(peps)
    order = np.argsort(ppl)
    for rank, i in enumerate(order, start=1):
        rows.append({"peptide_id": f"L{L:02d}_{rank:04d}", "length": L,
                     "sequence": peps[i], "perplexity": round(float(ppl[i]), 4),
                     "rank_in_length": rank})
    el = time.time() - t0
    frac = (lengths.index(L) + 1) / len(lengths)
    print(f"  length {L:2d}: {len(peps):4d} unique, best perplexity "
          f"{ppl[order[0]]:.3f}   [{el / 60:.1f} min, "
          f"~{el / frac * (1 - frac) / 60:.0f} min left]", flush=True)

df = pd.DataFrame(rows)

# ------------------------------------------------------------------ gate
if gate == "everything":
    df["selected"] = True
elif gate == "perplexity cutoff":
    df["selected"] = df["perplexity"] <= float(ppl_cutoff)
else:
    df["selected"] = df["rank_in_length"] <= int(keep_top_n)

n_sel = int(df["selected"].sum())
print(f"\ngenerated {len(df):,} peptides, {n_sel} selected for cofolding")
if n_sel == 0:
    print("!! nothing selected. Raise keep_top_n, or relax the perplexity cutoff -- "
          f"the best perplexity seen was {df['perplexity'].min():.3f}")
mins = n_sel * (3.0 if not E["kernels"] else 0.5)
print(f"   step 4 will take roughly {mins / 60:.1f} h" if mins > 90 else
      f"   step 4 will take roughly {mins:.0f} min")
if n_sel > E["n_cofold"]:
    print(f"!! that is more than the {E['n_cofold']} this GPU is comfortable with. "
          f"Step 4 has a max_designs cap you can use.")

print("\nbest 10 by perplexity, across all lengths "
      "(remember: not comparable between lengths)")
print(df.nsmallest(10, "perplexity")[
    ["peptide_id", "length", "sequence", "perplexity"]].to_string(index=False))

df.to_csv(f"{WORK}/master.csv", index=False)
df.to_csv(f"{OUT}/master.csv", index=False)
json.dump({"length_min": int(length_min), "length_max": int(length_max),
           "n_per_length": int(n_per_length), "top_k": int(top_k), "gate": gate,
           "keep_top_n": int(keep_top_n), "ppl_cutoff": float(ppl_cutoff),
           "precision": precision, "seed": int(seed),
           "receptor_len": len(SEQ), "hotspots": T["hotspots_raw"],
           "crop": T["crop"], "chain": T["chain"], "source": T["src"]},
          open(f"{OUT}/settings.json", "w"), indent=2)
print(f"\nmaster.csv written to My Drive/{E['drive_folder']}/{T['run']}/")
