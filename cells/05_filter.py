#@title 5 &middot; Filter { display-mode: "form" }
#@markdown Re-run this as often as you like. It refolds nothing: every number it
#@markdown needs is already in `master.csv` or in the saved structures, so even
#@markdown changing the hotspots here is free. Each design keeps a `failed`
#@markdown column naming the gates it missed.
#@markdown
#@markdown **Structure gates.** ipTM is the interface confidence for the
#@markdown receptor-peptide pair; ipAE is the interface error in angstroms and is the
#@markdown sharpest single number here.
min_iptm = 0.6  #@param {type:"number"}
max_ipae = 10.0  #@param {type:"number"}
min_pep_plddt = 0.7  #@param {type:"number"}
max_receptor_rmsd = 0.0  #@param {type:"number"}
#@markdown **Hotspot gates.** `min_hotspot_contacts` is how many of your residues the
#@markdown peptide must touch. `min_hotspot_fraction` is the share of its contacts
#@markdown landing on the patch around them — this is what removes a peptide that
#@markdown binds confidently to the wrong face.
min_hotspot_contacts = 2  #@param {type:"integer"}
min_hotspot_fraction = 0.6  #@param {type:"number"}
#@markdown The residues the peptide must actually have bound, in **your file's own
#@markdown numbering**, e.g. `40,99,107`. Independent of step 4's steering: change it
#@markdown and the contacts are re-measured from the structures already saved, so
#@markdown nothing is refolded. Leave empty to skip both hotspot gates.
hotspots = ""  #@param {type:"string"}
patch_radius = 10.0  #@param {type:"number"}
contact_cutoff = 4.5  #@param {type:"number"}
#@markdown **Sequence gates**, for whether you can actually make the peptide.
max_cys = 1  #@param {type:"integer"}
charge_min = -3.0  #@param {type:"number"}
charge_max = 5.0  #@param {type:"number"}
max_gravy = 1.0  #@param {type:"number"}
drop_liabilities = False  #@param {type:"boolean"}
rank_by = "iptm_pair"  #@param ["iptm_pair", "ipae", "confidence_score", "perplexity", "pep_plddt"]

import json, os, re, shutil
import gemmi
import numpy as np
import pandas as pd

E = json.load(open("/content/pepmlm/env.json"))
T = json.load(open("/content/pepmlm/target.json"))
WORK, OUT = E["work"], T["out"]
def recover(name):
    """A new runtime starts with an empty /content, but the tables live in Drive.
    Bring one back rather than making you regenerate it."""
    local, saved = f"{WORK}/{name}", f"{OUT}/{name}"
    if not os.path.isfile(local) and os.path.isfile(saved):
        os.makedirs(WORK, exist_ok=True)
        shutil.copyfile(saved, local)
        print(f"recovered {name} from My Drive/{E['drive_folder']}/{T['run']}/")
    return local


assert os.path.isfile(recover("master.csv")), (
    f"no master.csv, here or in My Drive/{E['drive_folder']}"
    f"/{T['run']}/ -- run step 3 to generate the peptides")
df = pd.read_csv(f"{WORK}/master.csv")
assert "cofolded" in df.columns, "run step 4 first"
d = df[df["cofolded"].eq(True)].copy()
assert len(d), "nothing has been cofolded yet"

for c in ("iptm", "iptm_pair", "ptm", "confidence_score", "complex_plddt",
          "complex_iplddt", "complex_ipde", "pep_plddt", "ipae", "n_contact_res",
          "hotspot_contacts", "hotspot_fraction", "receptor_rmsd"):
    if c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")

# plddt is reported on 0-1 by Boltz; tolerate a 0-100 file
for c in ("pep_plddt", "complex_plddt", "complex_iplddt"):
    if c in d.columns and d[c].max() > 1.5:
        d[c] = d[c] / 100.0

# ------------------------------------------------------------ hotspots
LAB = T["labels"]
_given = [x for x in hotspots.replace(" ", "").split(",") if x]
if _given:
    _ix = {lab: i + 1 for i, lab in enumerate(LAB)}
    _miss = [p for p in _given if p not in _ix]
    assert not _miss, (
        f"residue(s) {_miss} are not in the target step 2 built. Its numbering runs "
        f"{LAB[0]} .. {LAB[-1]}, and a crop in step 2 may have removed them.")
    HOT = [_ix[p] for p in _given]
    print("hotspots for this filter, overriding step 2:")
    for p, i in zip(_given, HOT):
        print(f"   {p:>6s}  ->  {T['seq'][i - 1]}{i}")
else:
    HOT = []
    print("no hotspots set here, so the two hotspot gates are skipped")

# ------------------------------------------------------------ re-measure geometry
# Done here rather than reused from step 4, so the hotspots, the patch radius and the
# contact cutoff can all be changed without refolding a single design.
PATCH = set()
_geom = T.get("geom") or T["template"]
if HOT and _geom and os.path.isfile(_geom):
    _t = gemmi.read_structure(_geom)
    _ca = {}
    for r in _t[0]["A"]:
        a = r.find_atom("CA", "*")
        if a:
            _ca[r.seqid.num] = np.array([a.pos.x, a.pos.y, a.pos.z])
    _h = np.array([_ca[i] for i in HOT if i in _ca])
    if len(_h):
        PATCH = {i for i, p in _ca.items()
                 if np.min(np.linalg.norm(_h - p, axis=1)) <= float(patch_radius)}


def measure(cif):
    st = gemmi.read_structure(cif)
    st.remove_hydrogens()
    ch = {c.name: c for c in st[0]}
    if "A" not in ch or "B" not in ch:
        return None
    ra, rid = [], []
    for r in ch["A"]:
        for a in r:
            ra.append([a.pos.x, a.pos.y, a.pos.z])
            rid.append(r.seqid.num)
    pb = [[a.pos.x, a.pos.y, a.pos.z] for r in ch["B"] for a in r]
    if not ra or not pb:
        return None
    ra, pb, rid = np.array(ra), np.array(pb), np.array(rid)
    dist = np.linalg.norm(ra[:, None, :] - pb[None, :, :], axis=-1)
    touched = sorted({int(x) for x in rid[(dist <= float(contact_cutoff)).any(axis=1)]})
    return {"n_contact_res": len(touched),
            "contact_residues": ";".join(str(x) for x in touched),
            "hotspot_contacts": len([i for i in HOT if i in touched]),
            "hotspot_fraction": (round(len([i for i in touched if i in PATCH])
                                       / len(touched), 3)
                                 if touched and PATCH else 0.0)}


for c in ("n_contact_res", "contact_residues", "hotspot_contacts",
          "hotspot_fraction"):
    if c not in d.columns:
        d[c] = pd.NA
d["contact_residues"] = d["contact_residues"].astype("object")
_done = 0
for i in d.index:
    _cif = f"{OUT}/structures/{d.at[i, 'peptide_id']}.cif"
    _m = measure(_cif) if os.path.isfile(_cif) else None
    if _m:
        for k, v in _m.items():
            d.at[i, k] = v
        _done += 1
if _done:
    print(f"re-measured {_done} design(s): contacts within {contact_cutoff} A, "
          f"{len(HOT)} hotspot(s), {len(PATCH)} residue(s) in the patch")
else:
    print("no saved structures found, so step 4's contact numbers are used as they "
          "are -- changing the hotspots or either radius here will do nothing")
for c in ("n_contact_res", "hotspot_contacts", "hotspot_fraction"):
    d[c] = pd.to_numeric(d[c], errors="coerce")

# ------------------------------------------------------------ sequence properties
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2,
          I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9,
          Y=-1.3, V=4.2)
LIAB = [("N-terminal Gln", lambda s: s.startswith("Q")),
        ("Asp-Pro (acid-labile)", lambda s: "DP" in s),
        ("Asn-Gly (deamidation)", lambda s: "NG" in s),
        ("Asp-Gly (isomerisation)", lambda s: "DG" in s),
        ("Met (oxidation)", lambda s: "M" in s),
        ("4+ identical in a row", lambda s: bool(re.search(r"(.)\1{3,}", s))),
        ("3+ Pro in a row", lambda s: "PPP" in s)]


def props(s):
    pos = s.count("K") + s.count("R") + 0.1 * s.count("H")
    flags = [n for n, f in LIAB if f(s)]
    return pd.Series({
        "n_cys": s.count("C"),
        "net_charge": round(pos - s.count("D") - s.count("E"), 1),
        "gravy": round(float(np.mean([KD[c] for c in s])), 3),
        "liabilities": "; ".join(flags),
        "n_liabilities": len(flags)})


# Everything below is recomputed, so drop any copy left by an earlier filter run.
# Without this, re-running step 5 on a master.csv it has already written appends a
# second n_cys, net_charge, gravy ... and every later lookup returns two columns.
d = d.drop(columns=[c for c in ("n_cys", "net_charge", "gravy", "liabilities",
                                "n_liabilities", "failed", "pass_all", "rank")
                    if c in d.columns])
d = pd.concat([d, d["sequence"].apply(props)], axis=1)
assert not d.columns.duplicated().any(), (
    f"duplicated columns: {sorted(set(d.columns[d.columns.duplicated()]))}")

# ------------------------------------------------------------ the gates
G = []
G.append(("iptm", d["iptm_pair"].astype(float) >= float(min_iptm)))
if "ipae" in d.columns:
    G.append(("ipae", d["ipae"].astype(float) <= float(max_ipae)))
if "pep_plddt" in d.columns:
    G.append(("plddt", d["pep_plddt"].astype(float) >= float(min_pep_plddt)))
if float(max_receptor_rmsd) > 0 and "receptor_rmsd" in d.columns:
    r = pd.to_numeric(d["receptor_rmsd"], errors="coerce")
    G.append(("receptor_moved", r.isna() | (r <= float(max_receptor_rmsd))))
if HOT:
    G.append(("hotspot_contacts",
              d["hotspot_contacts"].astype(float) >= int(min_hotspot_contacts)))
    G.append(("wrong_face",
              d["hotspot_fraction"].astype(float) >= float(min_hotspot_fraction)))
else:
    print("no hotspots set in this cell, so the two hotspot gates are skipped")
G.append(("cys", d["n_cys"] <= int(max_cys)))
G.append(("charge", d["net_charge"].between(float(charge_min), float(charge_max))))
G.append(("gravy", d["gravy"] <= float(max_gravy)))
if drop_liabilities:
    G.append(("liabilities", d["n_liabilities"] == 0))

failed = []
for i in d.index:
    failed.append(", ".join(n for n, m in G if not bool(m.loc[i])))
d["failed"] = failed
d["pass_all"] = d["failed"] == ""

print(f"{len(d)} cofolded designs\n")
print(f"{'gate':<20s} {'passing':>8s}")
print("-" * 30)
for n, m in G:
    print(f"{n:<20s} {int(m.sum()):>5d}/{len(d)}")
print("-" * 30)
print(f"{'ALL GATES':<20s} {int(d['pass_all'].sum()):>5d}/{len(d)}")

asc = rank_by in ("ipae", "perplexity")
final = d[d["pass_all"]].sort_values(rank_by, ascending=asc).reset_index(drop=True)
final.insert(0, "rank", np.arange(1, len(final) + 1))

show = [c for c in ["rank", "peptide_id", "length", "sequence", "perplexity",
                    "iptm_pair", "ipae", "pep_plddt", "hotspot_contacts",
                    "hotspot_fraction", "net_charge", "gravy", "n_cys",
                    "liabilities"] if c in final.columns]
if len(final):
    print(f"\n{len(final)} passing, ranked by {rank_by}:\n")
    print(final[show].head(25).to_string(index=False))
    shutil.rmtree(f"{OUT}/passing", ignore_errors=True)
    os.makedirs(f"{OUT}/passing", exist_ok=True)
    for _, r in final.iterrows():
        s = f"{OUT}/structures/{r['peptide_id']}.cif"
        if os.path.isfile(s):
            shutil.copyfile(s, f"{OUT}/passing/rank{int(r['rank']):03d}_"
                               f"{r['peptide_id']}.cif")
else:
    print("\nNothing passed. The gate table above shows where they died.")
    near = d.nlargest(min(10, len(d)), "iptm_pair")
    print("\nthe 10 most confident, and what each one failed:\n")
    print(near[[c for c in show if c in near.columns] + ["failed"]]
          .to_string(index=False))

final.to_csv(f"{OUT}/final.csv", index=False)
final.to_csv(f"{WORK}/final.csv", index=False)

# master.csv must keep EVERY generated peptide, so merge this cell's columns back into
# the full table. Writing `d` here would silently drop every peptide that was never
# selected for cofolding.
_full = df.set_index("peptide_id")
_sub = d.set_index("peptide_id")
for c in _sub.columns:
    if c not in _full.columns:
        _full[c] = pd.NA
    # Compare as plain objects: a column read back from CSV may have landed on a
    # narrower dtype than the value being written into it.
    _full[c] = _full[c].astype(object)
_full.update(_sub)
_full = _full.reset_index()
assert len(_full) == len(df), f"master.csv would lose rows ({len(_full)} vs {len(df)})"
_full.to_csv(f"{OUT}/master.csv", index=False)
_full.to_csv(f"{WORK}/master.csv", index=False)
json.dump({"min_iptm": float(min_iptm), "max_ipae": float(max_ipae),
           "min_pep_plddt": float(min_pep_plddt),
           "max_receptor_rmsd": float(max_receptor_rmsd),
           "min_hotspot_contacts": int(min_hotspot_contacts),
           "min_hotspot_fraction": float(min_hotspot_fraction),
           "max_cys": int(max_cys), "charge": [float(charge_min), float(charge_max)],
           "max_gravy": float(max_gravy), "drop_liabilities": bool(drop_liabilities),
           "rank_by": rank_by, "n_pass": int(len(final)),
           "hotspots_raw": hotspots.strip(), "hotspots_idx": HOT,
           "patch_radius": float(patch_radius),
           "contact_cutoff": float(contact_cutoff)},
          open(f"{OUT}/filter_settings.json", "w"), indent=2)
print(f"\nmaster.csv  {len(_full)} peptides generated, {len(d)} of them cofolded, "
      f"every metric, unfiltered")
print(f"final.csv   {len(final)} passing, ranked")
print(f"both in My Drive/{E['drive_folder']}/{T['run']}/")
