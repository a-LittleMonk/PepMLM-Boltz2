#@title 4 &middot; Cofold with Boltz-2 { display-mode: "form" }
#@markdown This is the slow step. It is resumable: re-running skips designs that are
#@markdown already done, so a Colab disconnect costs you one design.
#@markdown
#@markdown `sampling_steps` 50 is a screening setting. Boltz's published default is
#@markdown 200 — worth re-running on the finalists once you have some.
sampling_steps = 50  #@param {type:"integer"}
recycling_steps = 3  #@param {type:"integer"}
diffusion_samples = 1  #@param {type:"integer"}
#@markdown Cap on how many designs to attempt (0 = every one step 3 selected):
max_designs = 0  #@param {type:"integer"}
#@markdown **Hotspot steering.** The residues you want the peptide to bind, in your
#@markdown file's own numbering, e.g. `40,99,107`. This only aims the folding; what the
#@markdown peptide actually touched is judged in step 5. Leave empty to let Boltz put
#@markdown the peptide wherever it likes.
hotspots = ""  #@param {type:"string"}
#@markdown `force` turns the soft hint into a hard potential — leave it off, since a
#@markdown forced pose can look good without being real:
pocket_max_distance = 6.0  #@param {type:"number"}
pocket_force = False  #@param {type:"boolean"}
#@markdown Geometry measured afterwards. `patch_radius` defines the neighbourhood
#@markdown around your hotspots that counts as "the right face":
contact_cutoff = 4.5  #@param {type:"number"}
patch_radius = 10.0  #@param {type:"number"}
chunk_size = 25  #@param {type:"integer"}
#@markdown How many alignment sequences each design is folded against. Boltz's own
#@markdown help text says it subsamples to 1024 by default, but the flag that does it
#@markdown defaults to off, so without this it uses up to 8192 — several times the
#@markdown work for very little gain. Untick only if you want upstream's raw behaviour.
subsample_msa = True  #@param {type:"boolean"}
msa_depth = 1024  #@param {type:"integer"}
redo = False  #@param {type:"boolean"}

import glob, json, os, re, shutil, subprocess, sys, time
import gemmi
import numpy as np
import pandas as pd

E = json.load(open("/content/pepmlm/env.json"))
T = json.load(open("/content/pepmlm/target.json"))
WORK, OUT, SEQ = E["work"], T["out"], T["seq"]
BENV = {**os.environ, "PATH": f"{E['env']}/bin:" + os.environ.get("PATH", ""),
        "PYTHONPATH": "", "BOLTZ_CACHE": E["cache"], "HF_HOME": E["hf"]}

# Hotspots, in your own numbering, translated to positions in the sequence that
# step 2 built. They steer the folding only.
_given = [x for x in hotspots.replace(" ", "").split(",") if x]
_ix = {lab: i + 1 for i, lab in enumerate(T["labels"])}
_miss = [p for p in _given if p not in _ix]
assert not _miss, (
    f"residue(s) {_miss} are not in the target step 2 built. Its numbering runs "
    f"{T['labels'][0]} .. {T['labels'][-1]}, and a crop in step 2 may have removed them.")
HOT = [_ix[p] for p in _given]
if HOT:
    print("steering towards:")
    for p, i in zip(_given, HOT):
        print(f"   {p:>6s}  ->  {SEQ[i - 1]}{i}")
else:
    print("no hotspots: Boltz will place the peptide wherever it prefers")

df = pd.read_csv(f"{WORK}/master.csv")
todo = df[df["selected"]].copy()
if max_designs and len(todo) > int(max_designs):
    todo = todo.nsmallest(int(max_designs), "perplexity")
assert len(todo), "step 3 selected nothing to cofold"

YD, CD = f"{WORK}/yaml", f"{WORK}/cofold"
if redo:
    shutil.rmtree(YD, ignore_errors=True)
    shutil.rmtree(CD, ignore_errors=True)
os.makedirs(YD, exist_ok=True)
os.makedirs(CD, exist_ok=True)

# --------------------------------------------------------- the hotspot patch
# Receptor residues whose CA sits within patch_radius of any hotspot CA. A peptide
# that contacts the receptor only outside this patch is binding the wrong face.
_tpl = gemmi.read_structure(T["template"])
_ca = {}
for r in _tpl[0]["A"]:
    a = r.find_atom("CA", "*")
    if a:
        _ca[r.seqid.num] = np.array([a.pos.x, a.pos.y, a.pos.z])
if HOT:
    _h = np.array([_ca[i] for i in HOT if i in _ca])
    PATCH = {i for i, p in _ca.items()
             if np.min(np.linalg.norm(_h - p, axis=1)) <= float(patch_radius)}
else:
    PATCH = set()
print(f"{len(SEQ)} residues, {len(HOT)} hotspot(s), "
      f"{len(PATCH)} residues in the hotspot patch ({patch_radius} A)")

# --------------------------------------------------------- write the inputs
def write_yaml(path, pep):
    y = ["version: 1", "sequences:",
         "  - protein:", "      id: A", f"      sequence: {SEQ}",
         f"      msa: {T['msa']}",
         "  - protein:", "      id: B", f"      sequence: {pep}",
         "      msa: empty"]
    if T["template"]:
        y += ["templates:", f"  - cif: {T['template']}",
              "    chain_id: A", "    template_id: A"]
    if HOT:
        y += ["constraints:", "  - pocket:", "      binder: B",
              "      contacts: [" + ", ".join(f"[A, {i}]" for i in HOT) + "]",
              f"      max_distance: {float(pocket_max_distance)}",
              f"      force: {'true' if pocket_force else 'false'}"]
    open(path, "w").write("\n".join(y) + "\n")


ids = list(todo["peptide_id"])
chunks = [ids[i:i + int(chunk_size)] for i in range(0, len(ids), int(chunk_size))]
seqof = dict(zip(todo["peptide_id"], todo["sequence"]))
for ci, ch in enumerate(chunks):
    d = f"{YD}/c{ci:03d}"
    os.makedirs(d, exist_ok=True)
    for pid in ch:
        write_yaml(f"{d}/{pid}.yaml", seqof[pid])

# --------------------------------------------------------- measure one design
def interface(cif, L):
    """Contacts, the hotspot patch overlap, and how far the receptor moved."""
    st = gemmi.read_structure(cif)
    st.remove_hydrogens()
    m = st[0]
    rec = {c.name: c for c in m}
    A, B = rec.get("A"), rec.get("B")
    if A is None or B is None:
        return {}
    ra, rid = [], []
    for r in A:
        for a in r:
            if a.element != gemmi.Element("H"):
                ra.append([a.pos.x, a.pos.y, a.pos.z])
                rid.append(r.seqid.num)
    pb = [[a.pos.x, a.pos.y, a.pos.z] for r in B for a in r
          if a.element != gemmi.Element("H")]
    if not ra or not pb:
        return {}
    ra, pb, rid = np.array(ra), np.array(pb), np.array(rid)
    d = np.linalg.norm(ra[:, None, :] - pb[None, :, :], axis=-1)
    hit = rid[(d <= float(contact_cutoff)).any(axis=1)]
    touched = sorted(set(int(x) for x in hit))
    out = {"n_contact_res": len(touched),
           "contact_residues": ";".join(str(x) for x in touched[:40])}
    out["hotspot_contacts"] = len([i for i in HOT if i in touched])
    out["hotspot_fraction"] = (round(len([i for i in touched if i in PATCH])
                                     / len(touched), 3) if touched and PATCH else 0.0)
    # how much the receptor had to move relative to the free prediction
    out["receptor_rmsd"] = np.nan
    if os.path.isfile(T["apo"]):
        try:
            ap = gemmi.read_structure(T["apo"])
            p1, p2 = [], []
            apo_ca = {r.seqid.num: r.find_atom("CA", "*") for r in ap[0]["A"]}
            for r in A:
                a = r.find_atom("CA", "*")
                b = apo_ca.get(r.seqid.num)
                if a and b:
                    p1.append(a.pos)
                    p2.append(b.pos)
            if len(p1) > 10:
                out["receptor_rmsd"] = round(
                    gemmi.superpose_positions(p1, p2).rmsd, 3)
        except Exception:
            pass
    return out


def collect(pid, pred_dir, L):
    g = f"{pred_dir}/{pid}"
    cif = f"{g}/{pid}_model_0.cif"
    cj = f"{g}/confidence_{pid}_model_0.json"
    if not (os.path.isfile(cif) and os.path.isfile(cj)):
        return None
    c = json.load(open(cj))
    pair = c.get("pair_chains_iptm", {})
    try:
        ip = float(pair["0"]["1"])
    except Exception:
        ip = float(c.get("iptm", np.nan))
    row = {"cofolded": True,
           "iptm": round(float(c.get("iptm", np.nan)), 4),
           "iptm_pair": round(ip, 4),
           "ptm": round(float(c.get("ptm", np.nan)), 4),
           "confidence_score": round(float(c.get("confidence_score", np.nan)), 4),
           "complex_plddt": round(float(c.get("complex_plddt", np.nan)), 4),
           "complex_iplddt": round(float(c.get("complex_iplddt", np.nan)), 4),
           "complex_ipde": round(float(c.get("complex_ipde", np.nan)), 4)}
    pl = f"{g}/plddt_{pid}_model_0.npz"
    if os.path.isfile(pl):
        v = np.load(pl)["plddt"].reshape(-1).astype(float)
        if v.size and v.max() > 1.5:
            v = v / 100.0                      # some builds report pLDDT on 0-100
        if len(v) >= len(SEQ) + L:
            row["pep_plddt"] = round(float(np.mean(v[len(SEQ):len(SEQ) + L])), 4)
    pa = f"{g}/pae_{pid}_model_0.npz"
    if os.path.isfile(pa):
        p = np.load(pa)["pae"]
        p = p[0] if p.ndim == 3 else p
        if p.shape[0] >= len(SEQ) + L:
            a, b = slice(0, len(SEQ)), slice(len(SEQ), len(SEQ) + L)
            row["ipae"] = round(float((p[a, b].mean() + p[b, a].mean()) / 2), 4)
    row.update(interface(cif, L))
    row["cif_name"] = f"{pid}.cif"
    shutil.copyfile(cif, f"{OUT}/structures/{pid}.cif")
    return row


# --------------------------------------------------------- run
_PROG = re.compile(r"(Predicting|Generating|Processing).*?(\d+)%")
_NUM = ["iptm", "iptm_pair", "ptm", "confidence_score", "complex_plddt",
        "complex_iplddt", "complex_ipde", "pep_plddt", "ipae", "n_contact_res",
        "hotspot_contacts", "hotspot_fraction", "receptor_rmsd"]


def tidy(d):
    """Columns created empty and filled in later stay a generic object dtype, which
    pandas will not sort or rank on. Give them their real types back."""
    for c in _NUM:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "cofolded" in d.columns:
        d["cofolded"] = d["cofolded"].eq(True)
    return d
done_before = int(df.get("cofolded", pd.Series(dtype=bool)).fillna(False).sum()) \
    if "cofolded" in df else 0
_depth = max(sum(1 for _ in open(T["msa"])) - 1, 0)
print(f"\n{len(ids)} design(s) in {len(chunks)} chunk(s) of {chunk_size}; "
      f"{done_before} already done in master.csv")
print(f"alignment depth {_depth:,}"
      + (f", subsampled to {int(msa_depth):,}" if subsample_msa and _depth > msa_depth
         else "")
      + f"   |   {int(sampling_steps)} sampling steps, "
        f"{int(recycling_steps)} recycles, {int(diffusion_samples)} sample(s)")
print("the progress bar only moves when a design finishes, so it sits at 0% through "
      "the first one\n")

t0, finished = time.time(), 0
for ci, ch in enumerate(chunks):
    d, od = f"{YD}/c{ci:03d}", f"{CD}/c{ci:03d}"
    pred = f"{od}/boltz_results_c{ci:03d}/predictions"
    have = sum(1 for p in ch
               if os.path.isfile(f"{pred}/{p}/confidence_{p}_model_0.json"))
    if have == len(ch):
        print(f"chunk {ci + 1}/{len(chunks)}: already complete")
    else:
        cmd = [f"{E['env']}/bin/boltz", "predict", d, "--out_dir", od,
               "--cache", E["cache"], "--recycling_steps", str(int(recycling_steps)),
               "--sampling_steps", str(int(sampling_steps)),
               "--diffusion_samples", str(int(diffusion_samples)),
               "--output_format", "mmcif", "--write_full_pae",
               "--num_workers", "2", "--preprocessing-threads", "2"]
        if subsample_msa:
            cmd += ["--subsample_msa", "--num_subsampled_msa", str(int(msa_depth))]
        if pocket_force:
            cmd.append("--use_potentials")
        if not E["kernels"]:
            cmd.append("--no_kernels")
        print(f"chunk {ci + 1}/{len(chunks)}: {len(ch) - have} to run ...", flush=True)
        p = subprocess.Popen(cmd, env=BENV, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, bufsize=1)
        tail, w, refused, shown = [], 0, [], 0
        for line in p.stdout:
            tail.append(line.rstrip())
            tail[:] = tail[-60:]
            s = line.strip()
            # tqdm writes its bar with a bare \r, so a failure message often arrives
            # appended to the end of a progress line. Check for it first.
            if "Failed to process" in s:
                refused.append(s.split("Error:")[-1].strip().rstrip(".") or s)
            elif _PROG.search(s):
                txt = f"    {s[:110]}"
                w = max(w, len(txt))
                sys.stdout.write("\r" + txt.ljust(w))
                sys.stdout.flush()
            elif "Traceback" in s or s.split(":")[0].endswith("Error"):
                pass                      # already counted above, and very repetitive
            elif "rror" in s and shown < 5:
                shown += 1
                sys.stdout.write("\n    " + s + "\n")
        p.wait()
        sys.stdout.write("\n")
        if refused:
            from collections import Counter
            print(f"    Boltz refused {len(refused)} of {len(ch)} input file(s):")
            for why, n in Counter(refused).most_common(3):
                print(f"      {n} x  {why}")
        if p.returncode != 0:
            print("\n".join(tail))
            raise RuntimeError(
                f"boltz failed on chunk {ci} (exit {p.returncode}). If this says out "
                f"of memory, lower chunk_size or crop the receptor further.")

    got = {}
    for pid in ch:
        r = collect(pid, pred, int(df.loc[df.peptide_id == pid, "length"].iloc[0]))
        if r:
            got[pid] = r
    if not got:
        # Stop here rather than march through every chunk producing nothing.
        print("\n".join(tail[-25:]) if "tail" in dir() else "")
        raise RuntimeError(
            f"none of the {len(ch)} designs in chunk {ci} produced output.\n"
            f"  Boltz's own last lines are above. The inputs it read are in {d},\n"
            f"  and a single one can be retried by hand with:\n"
            f"    !{E['env']}/bin/boltz predict {d}/{ch[0]}.yaml --out_dir /content/try "
            f"--cache {E['cache']}")
    for pid, r in got.items():
        for k, v in r.items():
            if k not in df.columns:
                df[k] = pd.NA
            df.loc[df.peptide_id == pid, k] = v
    df = tidy(df)
    df.to_csv(f"{WORK}/master.csv", index=False)
    df.to_csv(f"{OUT}/master.csv", index=False)
    finished += len(got)
    el = time.time() - t0
    rate = el / max(finished, 1)
    left = (len(ids) - (ci + 1) * int(chunk_size))
    print(f"    {len(got)}/{len(ch)} parsed   {rate:.0f} s/design   "
          f"{el / 60:.0f} min elapsed"
          + (f", ~{max(left, 0) * rate / 60:.0f} min left" if left > 0 else ""))

df = tidy(df)
ok = df["cofolded"] if "cofolded" in df else pd.Series(False, index=df.index)
print(f"\ncofolded {int(ok.sum())} of {len(ids)} in {(time.time() - t0) / 60:.1f} min")
if int(ok.sum()) < len(ids):
    print("   some designs produced no output; re-running this cell retries only those")
cols = [c for c in ["peptide_id", "length", "sequence", "perplexity", "iptm_pair",
                    "ipae", "pep_plddt", "hotspot_contacts", "hotspot_fraction"]
        if c in df.columns]
if int(ok.sum()) and "iptm_pair" in df.columns:
    print("\nbest 10 by interface ipTM:")
    print(df[ok].nlargest(10, "iptm_pair")[cols].to_string(index=False))
print(f"\nstructures -> My Drive/{E['drive_folder']}/{T['run']}/structures/")
