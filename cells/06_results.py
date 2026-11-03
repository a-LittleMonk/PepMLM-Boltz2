#@title 6 &middot; Look at a result { display-mode: "form" }
which = "passing (final.csv)"  #@param ["passing (final.csv)", "any cofolded (master.csv)"]
#@markdown Which row to show: rank 1 is the best by the ranking you chose in step 5.
rank = 1  #@param {type:"integer"}
colour_by = "pLDDT"  #@param ["pLDDT", "plain"]
show_contacts = True  #@param {type:"boolean"}

import json, os
import pandas as pd
import py3Dmol

E = json.load(open("/content/pepmlm/env.json"))
T = json.load(open("/content/pepmlm/target.json"))
WORK, OUT, HOT = E["work"], T["out"], T["hotspots_idx"]

if which.startswith("passing"):
    d = pd.read_csv(f"{WORK}/final.csv")
    assert len(d), "final.csv is empty -- loosen the gates in step 5"
else:
    d = pd.read_csv(f"{WORK}/master.csv")
    d = d[d["cofolded"].eq(True)].copy()
    d["iptm_pair"] = pd.to_numeric(d["iptm_pair"], errors="coerce")
    d = d.sort_values("iptm_pair", ascending=False).reset_index(drop=True)
    d.insert(0, "rank", range(1, len(d) + 1))

i = max(1, min(int(rank), len(d))) - 1
r = d.iloc[i]
cif = f"{OUT}/structures/{r['peptide_id']}.cif"
assert os.path.isfile(cif), f"no structure saved for {r['peptide_id']}"

print(f"rank {int(r.get('rank', i + 1))} of {len(d)}   {r['peptide_id']}")
print(f"  sequence        {r['sequence']}   ({int(r['length'])} aa)")
print(f"  perplexity      {r['perplexity']:.3f}")
print(f"  interface ipTM  {r['iptm_pair']:.3f}"
      + (f"   ipAE {r['ipae']:.2f} A" if "ipae" in r and pd.notna(r["ipae"]) else ""))
if "pep_plddt" in r and pd.notna(r["pep_plddt"]):
    print(f"  peptide pLDDT   {float(r['pep_plddt']):.3f}")
if HOT:
    print(f"  hotspots hit    {int(r['hotspot_contacts'])} of {len(HOT)}"
          f"   on-patch fraction {float(r['hotspot_fraction']):.2f}")
if "contact_residues" in r and isinstance(r["contact_residues"], str):
    lab = T["labels"]
    got = [int(x) for x in r["contact_residues"].split(";") if x]
    print(f"  touches         "
          + ", ".join(f"{T['seq'][j - 1]}{lab[j - 1]}" for j in got[:18])
          + (" ..." if len(got) > 18 else ""))
    print("                  (your own numbering)")
if "liabilities" in r and isinstance(r["liabilities"], str) and r["liabilities"]:
    print(f"  watch out for   {r['liabilities']}")

v = py3Dmol.view(width=820, height=520)
v.addModel(open(cif).read(), "cif")
v.setStyle({"chain": "A"}, {"cartoon": {"color": "grey", "opacity": 0.8}})
if colour_by == "pLDDT":
    v.setStyle({"chain": "B"}, {"cartoon": {"colorscheme": {"prop": "b",
                "gradient": "roygb", "min": 0.5, "max": 1.0}}})
    v.addStyle({"chain": "B"}, {"stick": {"radius": 0.16}})
else:
    v.setStyle({"chain": "B"}, {"cartoon": {"color": "orange"},
                                "stick": {"radius": 0.16}})
if HOT:
    v.addStyle({"chain": "A", "resi": [str(x) for x in HOT]},
               {"stick": {"colorscheme": "pinkCarbon", "radius": 0.28}})
if show_contacts and "contact_residues" in r and isinstance(r["contact_residues"], str):
    cr = [x for x in r["contact_residues"].split(";") if x]
    v.addStyle({"chain": "A", "resi": cr},
               {"cartoon": {"color": "lightblue"}})
v.zoomTo({"chain": "B"})
v.show()
msg = "grey = target, "
msg += "peptide red->blue by pLDDT" if colour_by == "pLDDT" else "orange = peptide"
print("\n" + msg + ", pink sticks = your hotspots"
      + (", pale blue = residues the peptide touches" if show_contacts else ""))
