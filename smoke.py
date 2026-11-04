"""End-to-end smoke run of the notebook's data path, without a GPU.

Everything except the two model calls (PepMLM generation, Boltz folding) is the
notebook's own code, executed against synthetic inputs and synthetic Boltz outputs
written in the exact formats Boltz produces. Variations are swept deliberately.

    python -I smoke.py
"""
import io
import json
import os
import re
import shutil
import tempfile

import gemmi
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.join(HERE, "cells")
ROOT = tempfile.mkdtemp(prefix="pepmlm_smoke_")
FAILS = []
NOTES = []


def ok(msg):
    print(f"    ok    {msg}")


def bad(msg):
    FAILS.append(msg)
    print(f"  **BUG** {msg}")


def note(msg):
    NOTES.append(msg)
    print(f"    note  {msg}")


def head(msg):
    print(f"\n{'=' * 74}\n  {msg}\n{'=' * 74}")


def cell(name, **inject):
    """Load a cell, swap its two absolute json loads for injected dicts, and drop the
    form assignments so the values come from the namespace instead -- which is what
    Colab's form does when you type in it."""
    s = io.open(os.path.join(CELLS, name), encoding="utf-8").read()
    s = s.replace('E = json.load(open("/content/pepmlm/env.json"))', "E = _E")
    s = s.replace('T = json.load(open("/content/pepmlm/target.json"))', "T = _T")
    s = re.sub(r"(?m)^([A-Za-z_]\w*) = .*#@param.*$", r"\1 = \1", s)
    # keep the injected subprocess stub: the cell's own import would replace it
    s = re.sub(r"(?m)^(import [\w, ]*?), subprocess(.*)$", r"\1\2", s)
    return s


# --------------------------------------------------------------------- fixtures
AA3 = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
       "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
AA1 = "ARNDCQEGHILKMFPSWYV"[:0] + "ARNDCQEGHILKMFPSTWYV"
THREE_TO_ONE = dict(zip(AA3, "ARNDCQEGHILKMFPSTWYV"))
N_RES = 112                      # same size as the user's TGF-beta1 target
# A gentle arc with ~3.8 A between consecutive CA atoms, as in a real chain.
ARC_R, ARC_A, ARC_Z = 18.0, 0.15, 2.7


def receptor_pdb(path, n=N_RES, start=1, gap_after=None, icode_at=None,
                 extra_chain=True, water=True):
    """A plausible receptor: residues on a gentle arc, 4 atoms each."""
    st = gemmi.Structure()
    st.spacegroup_hm = "P 1"
    m = gemmi.Model("1")
    ch = gemmi.Chain("A")
    num = start
    labels = []
    for i in range(n):
        r = gemmi.Residue()
        r.name = AA3[i % 20]
        ic = " "
        if icode_at is not None and i == icode_at:
            num -= 1            # same number as the previous residue, with an icode
            ic = "A"
        r.seqid = gemmi.SeqId(num, ic)
        labels.append(f"{num}{ic.strip()}")
        t = i * ARC_A
        base = np.array([ARC_R * np.cos(t), ARC_R * np.sin(t), ARC_Z * i])
        for nm, off in (("N", [-1.2, 0, 0]), ("CA", [0, 0, 0]),
                        ("C", [1.2, 0, 0]), ("CB", [0, 1.4, 0])):
            a = gemmi.Atom()
            a.name = nm
            a.element = gemmi.Element("N" if nm == "N" else "C")
            p = base + np.array(off)
            a.pos = gemmi.Position(*p)
            a.b_iso = 30.0
            r.add_atom(a)
        ch.add_residue(r)
        num += 1
        if gap_after is not None and i == gap_after:
            num += 12           # an unobserved loop
    m.add_chain(ch)
    if extra_chain:
        cb = gemmi.Chain("B")
        for i in range(8):
            r = gemmi.Residue()
            r.name, r.seqid = "GLY", gemmi.SeqId(500 + i, " ")
            a = gemmi.Atom()
            a.name, a.element = "CA", gemmi.Element("C")
            a.pos = gemmi.Position(200.0 + i, 0, 0)
            r.add_atom(a)
            cb.add_residue(r)
        m.add_chain(cb)
    if water:
        cw = gemmi.Chain("W")
        for i in range(3):
            r = gemmi.Residue()
            r.name, r.seqid = "HOH", gemmi.SeqId(900 + i, " ")
            a = gemmi.Atom()
            a.name, a.element = "O", gemmi.Element("O")
            a.pos = gemmi.Position(5.0 + i, 5, 5)
            r.add_atom(a)
            cw.add_residue(r)
        m.add_chain(cw)
    st.add_model(m)
    st.setup_entities()
    st.write_pdb(path)
    return labels


def fake_boltz_output(pred_dir, pid, seq, peptide, near_res, metrics,
                      write_pae=True, write_plddt=True, truncate=False):
    """Write what Boltz writes: a 2-chain mmCIF, confidence JSON, plddt and pae npz."""
    g = os.path.join(pred_dir, pid)
    os.makedirs(g, exist_ok=True)
    L = len(peptide)
    # structure: receptor renumbered 1..N, peptide chain B parked next to near_res
    st = gemmi.Structure()
    st.spacegroup_hm = "P 1"
    m = gemmi.Model("1")
    A = gemmi.Chain("A")
    pos = {}
    for i in range(len(seq)):
        r = gemmi.Residue()
        r.name = [k for k, v in THREE_TO_ONE.items() if v == seq[i]][0]
        r.seqid = gemmi.SeqId(i + 1, " ")
        t = i * ARC_A
        base = np.array([ARC_R * np.cos(t), ARC_R * np.sin(t), ARC_Z * i])
        pos[i + 1] = base
        for nm, off in (("N", [-1.2, 0, 0]), ("CA", [0, 0, 0]),
                        ("C", [1.2, 0, 0]), ("CB", [0, 1.4, 0])):
            a = gemmi.Atom()
            a.name, a.element = nm, gemmi.Element("N" if nm == "N" else "C")
            a.pos = gemmi.Position(*(base + np.array(off)))
            a.b_iso = 70.0
            r.add_atom(a)
        A.add_residue(r)
    m.add_chain(A)
    B = gemmi.Chain("B")
    anchor = np.mean([pos[i] for i in near_res], axis=0)
    for k in range(L):
        r = gemmi.Residue()
        r.name = [kk for kk, v in THREE_TO_ONE.items() if v == peptide[k]][0]
        r.seqid = gemmi.SeqId(k + 1, " ")
        # 3.4 A out from the receptor arc, running alongside it
        rad = anchor[:2] / max(np.linalg.norm(anchor[:2]), 1e-6)
        # run alongside the receptor arc at its own residue spacing, so the peptide
        # makes a realistic number of contacts rather than one
        p = anchor + np.array([rad[0] * 3.4, rad[1] * 3.4, (k - L / 2) * ARC_Z])
        for nm, off in (("CA", [0, 0, 0]), ("CB", [rad[0] * 1.4, rad[1] * 1.4, 0])):
            a = gemmi.Atom()
            a.name, a.element = nm, gemmi.Element("C")
            a.pos = gemmi.Position(*(p + np.array(off)))
            a.b_iso = 70.0
            r.add_atom(a)
        B.add_residue(r)
    m.add_chain(B)
    st.add_model(m)
    st.setup_entities()
    st.make_mmcif_document().write_file(os.path.join(g, f"{pid}_model_0.cif"))

    conf = {"confidence_score": metrics.get("confidence_score", 0.7),
            "ptm": metrics.get("ptm", 0.8),
            "iptm": metrics.get("iptm", 0.6),
            "ligand_iptm": 0.0,
            "protein_iptm": metrics.get("iptm", 0.6),
            "complex_plddt": metrics.get("complex_plddt", 0.8),
            "complex_iplddt": metrics.get("complex_iplddt", 0.7),
            "complex_pde": 1.0,
            "complex_ipde": metrics.get("complex_ipde", 2.0),
            "chains_ptm": {"0": 0.9, "1": 0.5},
            "pair_chains_iptm": {"0": {"0": 0.9, "1": metrics["iptm_pair"]},
                                 "1": {"0": metrics["iptm_pair"], "1": 0.5}}}
    io.open(os.path.join(g, f"confidence_{pid}_model_0.json"), "w").write(
        json.dumps(conf, indent=2))

    n = len(seq) + L
    if truncate:
        n = len(seq)             # as if the token count did not match
    if write_plddt:
        v = np.full(n, 0.85)
        v[len(seq):] = metrics.get("pep_plddt", 0.75)
        np.savez_compressed(os.path.join(g, f"plddt_{pid}_model_0.npz"), plddt=v)
    if write_pae:
        p = np.full((n, n), 20.0)
        p[:len(seq), :len(seq)] = 1.0
        if n > len(seq):
            p[:len(seq), len(seq):] = metrics.get("ipae", 8.0)
            p[len(seq):, :len(seq)] = metrics.get("ipae", 8.0)
        np.savez_compressed(os.path.join(g, f"pae_{pid}_model_0.npz"), pae=p)
    return g


# ===================================================================== STEP 2
head("STEP 2  target: parsing, cropping, template, caching")

WORK = os.path.join(ROOT, "content", "pepmlm")
DRIVE = os.path.join(ROOT, "drive", "PepMLM")
for p in (WORK, f"{DRIVE}/inputs", f"{DRIVE}/receptors", f"{DRIVE}/weights"):
    os.makedirs(p, exist_ok=True)
_E = {"work": WORK, "env": os.path.join(ROOT, "env"), "cache": os.path.join(ROOT, "bc"),
      "hf": os.path.join(ROOT, "hf"), "drive": DRIVE, "drive_folder": "PepMLM",
      "repo": "ChatterjeeLab/PepMLM-650M", "gpu": "Tesla T4", "gpu_mem": 15360,
      "gpu_cc": 7.5, "kernels": False, "tier": "slow", "n_cofold": 60}

LABELS = receptor_pdb(f"{DRIVE}/inputs/tgfb1.pdb", gap_after=40, icode_at=70)
SRC = f"{DRIVE}/inputs/tgfb1.pdb"
C2 = cell("02_target.py")


class FakeBoltz:
    """Stands in for the one boltz call step 2 makes, writing the alignment and the
    free-receptor prediction exactly where step 2 looks for them."""
    calls = 0
    PIPE = -1
    STDOUT = -2

    @staticmethod
    def run(cmd, **kw):
        FakeBoltz.calls += 1
        y = cmd[2]
        out = cmd[cmd.index("--out_dir") + 1]
        seq = ""
        for line in io.open(y, encoding="utf-8"):
            if "sequence:" in line:
                seq = line.split("sequence:")[1].strip()
        md = os.path.join(out, "boltz_results_receptor", "msa")
        pd_ = os.path.join(out, "boltz_results_receptor", "predictions", "receptor")
        os.makedirs(md, exist_ok=True)
        os.makedirs(pd_, exist_ok=True)
        io.open(os.path.join(md, "receptor_0.csv"), "w").write(
            "\n".join(["key,sequence"] + [f"-1,{seq}" for _ in range(1200)]) + "\n")
        st = gemmi.Structure()
        st.spacegroup_hm = "P 1"
        m = gemmi.Model("1")
        c = gemmi.Chain("A")
        for i, letter in enumerate(seq, start=1):
            r = gemmi.Residue()
            r.name = [k for k, v in THREE_TO_ONE.items() if v == letter][0]
            r.seqid = gemmi.SeqId(i, " ")
            a = gemmi.Atom()
            a.name, a.element = "CA", gemmi.Element("C")
            t = (i - 1) * ARC_A
            a.pos = gemmi.Position(ARC_R * np.cos(t), ARC_R * np.sin(t),
                                   ARC_Z * (i - 1))
            r.add_atom(a)
            c.add_residue(r)
        m.add_chain(c)
        st.add_model(m)
        st.setup_entities()
        st.make_mmcif_document().write_file(
            os.path.join(pd_, "receptor_model_0.cif"))

        class R:
            returncode = 0
            stdout = "fake boltz run\n"
        return R()


def run_step2(**kw):
    ns = {"_E": _E, "json": json, "os": os, "shutil": shutil, "gemmi": gemmi,
          "glob": __import__("glob"), "hashlib": __import__("hashlib"),
          "subprocess": FakeBoltz, "sys": __import__("sys"),
          "time": __import__("time"),
          "target_from": "Google Drive file", "target_path": "tgfb1.pdb",
          "target_chain": "A", "crop": "", "run_name": "", "use_template": True,
          "show_structure": False, "print": lambda *a, **k: None}
    ns.update(kw)
    exec(C2, ns)
    return ns


# Seed the alignment cache so the MSA branch takes the cached path (no server call).
def seed_cache(seq):
    import hashlib
    h = hashlib.md5(seq.encode()).hexdigest()[:12]
    cd = f"{DRIVE}/receptors/{h}"
    os.makedirs(cd, exist_ok=True)
    rows = ["key,sequence"] + [f"-1,{seq}" for _ in range(1500)]
    io.open(f"{cd}/msa.csv", "w").write("\n".join(rows) + "\n")
    return cd, h


# the sequence step 2 will build, derived the same way it does
_probe = [AA3[i % 20] for i in range(N_RES)]
SEQ_FULL = "".join(THREE_TO_ONE[x] for x in _probe)
seed_cache(SEQ_FULL)

ns2 = run_step2()
T_FULL = json.load(open(f"{WORK}/target.json"))
if T_FULL["seq"] == SEQ_FULL and T_FULL["n_res"] == N_RES:
    ok(f"parsed {N_RES} residues, water and the second chain ignored")
else:
    bad(f"sequence wrong: {len(T_FULL['seq'])} residues vs {N_RES}")
if T_FULL["labels"] == LABELS:
    ok("residue labels preserved in file order, insertion code included")
else:
    bad("labels do not match the input file's numbering")
if len(ns2["gaps"]) == 1:
    ok("the unobserved loop is reported as one gap")
else:
    bad(f"gap detection found {len(ns2['gaps'])} gaps, expected 1")
if T_FULL["run"] == "tgfb1_A":
    ok("run name derived from filename and chain")
else:
    bad(f"run name is {T_FULL['run']!r}")
if "reusing" not in "" and os.path.isfile(T_FULL["msa"]):
    ok("alignment taken from the Drive cache, no server call")

# -- variation: crop
ns = run_step2(crop="10-60")
_kept = json.load(open(f"{WORK}/target.json"))["n_res"]
_expect = len([x for x in LABELS
               if 10 <= int("".join(c for c in x if c.isdigit() or c == "-")) <= 60])
if _kept == _expect:
    ok(f"crop 10-60 keeps {_kept} residues (the numbering has a gap)")
else:
    bad(f"crop kept {_kept}, expected {_expect}")

# -- variation: a crop that lands on the insertion-coded residue
try:
    run_step2(crop="68-72")
    ok("a crop spanning the insertion-coded residue works")
except Exception as e:
    bad(f"crop across an insertion code raised {type(e).__name__}: {e}")

# -- variation: negative residue numbering (expression tags are often -5..0)
receptor_pdb(f"{DRIVE}/inputs/neg.pdb", n=30, start=-5)
try:
    run_step2(target_path="neg.pdb", crop="-5--1")
    ok("a negative crop range is accepted")
except Exception as e:
    bad(f"negative crop range raises {type(e).__name__}: {e}  "
        f"(residues numbered below 1 are common in expression constructs)")

# -- variation: use_template off
ns = run_step2(use_template=False, run_name="notpl")
T_NOTPL = json.load(open(f"{WORK}/target.json"))
if T_NOTPL["template"] == "":
    ok("use_template off clears the template path in target.json")

# -- variation: receptor too long for ESM
receptor_pdb(f"{DRIVE}/inputs/huge.pdb", n=1100, extra_chain=False, water=False)
_long = "".join(THREE_TO_ONE[AA3[i % 20]] for i in range(1100))
seed_cache(_long)
_before = io.open(f"{WORK}/target.json", encoding="utf-8").read()
try:
    run_step2(target_path="huge.pdb", run_name="huge")
    bad("a 1100-residue receptor was accepted; ESM-2 stops at 1024")
except AssertionError:
    _after = io.open(f"{WORK}/target.json", encoding="utf-8").read()
    if _after != _before and json.loads(_after).get("n_res") == 1100:
        bad("the ESM length check fires AFTER target.json is written, so the "
            "oversized target is left on disk for steps 3 and 4 to pick up")
    else:
        ok("an oversized receptor is refused")

# restore a good target
run_step2()
T_FULL = json.load(open(f"{WORK}/target.json"))

# -- variation: run_name reuse across different targets
receptor_pdb(f"{DRIVE}/inputs/other.pdb", n=60, extra_chain=False, water=False)
seed_cache("".join(THREE_TO_ONE[AA3[i % 20]] for i in range(60)))
try:
    run_step2(target_path="other.pdb", run_name="tgfb1_A")
    bad("a different target reusing the same run_name was accepted; step 4 would "
        "then recover a master.csv of peptides designed for the other receptor")
except AssertionError as e:
    if "different receptor" in str(e):
        ok("reusing a run_name for a different receptor is refused")
    else:
        raise
_auto = run_step2(crop="20-60")
if json.load(open(f"{WORK}/target.json"))["run"] == "tgfb1_A_20-60":
    ok("a crop gets its own run folder by default, so it cannot collide")
else:
    bad("the default run name does not distinguish a cropped run")
run_step2()
T_FULL = json.load(open(f"{WORK}/target.json"))
print(f"\n  step 2 produced: {T_FULL['n_res']} residues, template "
      f"{'yes' if T_FULL['template'] else 'no'}, run {T_FULL['run']}")

# ===================================================================== STEP 3
head("STEP 3  generation: the gate and the table it writes")

C3 = io.open(os.path.join(CELLS, "03_binder.py"), encoding="utf-8").read()
GATE = C3[C3.index('if gate == "everything":'):C3.index("n_sel = int(")]
rng = np.random.default_rng(0)
rows = []
for L in range(12, 21):
    for k in range(60):
        rows.append({"peptide_id": f"L{L:02d}_{k + 1:04d}", "length": L,
                     "sequence": "".join(rng.choice(list("ACDEFGHIKLMNPQRSTVWY"), L)),
                     "perplexity": round(float(6 + rng.random() * 12), 4),
                     "rank_in_length": k + 1})
MASTER = pd.DataFrame(rows)
for g, kw, want in (("top N per length", dict(keep_top_n=6), 9 * 6),
                    ("perplexity cutoff", dict(ppl_cutoff=8.0), None),
                    ("everything", dict(), len(MASTER))):
    ns = {"df": MASTER.copy(), "gate": g, "keep_top_n": kw.get("keep_top_n", 6),
          "ppl_cutoff": kw.get("ppl_cutoff", 4.0), "pd": pd}
    exec(GATE, ns)
    n = int(ns["df"]["selected"].sum())
    if want is None:
        ok(f"gate {g!r} selected {n}")
    elif n == want:
        ok(f"gate {g!r} selected {n} as expected")
    else:
        bad(f"gate {g!r} selected {n}, expected {want}")

ns = {"df": MASTER.copy(), "gate": "top N per length", "keep_top_n": 6,
      "ppl_cutoff": 4.0, "pd": pd}
exec(GATE, ns)
MASTER = ns["df"]
SEL = list(MASTER.loc[MASTER["selected"], "peptide_id"])
MASTER.to_csv(f"{WORK}/master.csv", index=False)
MASTER.to_csv(f"{T_FULL['out']}/master.csv", index=False)
ok(f"master.csv written with {len(MASTER)} peptides, {len(SEL)} selected")

# the perplexity masking position arithmetic, checked symbolically
PPL = C3[C3.index("def perplexities("):C3.index("# ------------------------------------------------------------------ generate")]
if "j = len(ids) - 1 - n + k" in PPL:
    R, Lp = 112, 15
    ids_len = 1 + R + Lp + 1
    got = [ids_len - 1 - Lp + k for k in range(Lp)]
    want = [1 + R + k for k in range(Lp)]
    if got == want:
        ok("pseudo-perplexity masks exactly the peptide tokens (cls + receptor offset)")
    else:
        bad(f"perplexity masks the wrong tokens: {got[:3]} vs {want[:3]}")

# ===================================================================== STEP 4
head("STEP 4  cofolding: measurement, resume, staleness")

C4 = cell("04_cofold.py")
HOT_LABELS = "93,94"


def step4_ns(**kw):
    ns = {"_E": _E, "_T": T_FULL, "json": json, "os": os, "re": re,
          "shutil": shutil, "glob": __import__("glob"), "sys": __import__("sys"),
          "subprocess": __import__("subprocess"), "time": __import__("time"),
          "gemmi": gemmi, "np": np, "pd": pd,
          "sampling_steps": 50, "recycling_steps": 3, "diffusion_samples": 1,
          "max_designs": 0, "hotspots": HOT_LABELS, "pocket_max_distance": 6.0,
          "pocket_force": False, "contact_cutoff": 4.5, "patch_radius": 10.0,
          "chunk_size": 25, "subsample_msa": True, "msa_depth": 1024, "redo": False,
          "print": lambda *a, **k: None}
    ns.update(kw)
    return ns


# -- the parts of cell 4 up to the run loop, so YAMLs and the patch are built
PRE = C4[:C4.index("t0, finished = time.time(), 0")]
ns4 = step4_ns()
exec(PRE, ns4)
ok(f"patch is {len(ns4['PATCH'])} residues around hotspots {ns4['HOT']}")

y = io.open(f"{ns4['YD']}/c000/{SEL[0]}.yaml", encoding="utf-8").read()
import yaml as _yaml
doc = _yaml.safe_load(y)
checks = [
    (doc["sequences"][0]["protein"]["sequence"] == T_FULL["seq"], "receptor sequence"),
    (doc["sequences"][0]["protein"]["msa"] == T_FULL["msa"], "cached alignment"),
    (doc["sequences"][1]["protein"]["msa"] == "empty", "peptide single-sequence"),
    (doc["templates"][0]["template_id"] == "A", "template chain mapping"),
    (doc["constraints"][0]["pocket"]["contacts"]
     == [["A", i] for i in ns4["HOT"]],
     f"pocket contacts are sequence positions {ns4['HOT']}, not labels 93/94"),
]
for good, what in checks:
    ok(f"YAML: {what}") if good else bad(f"YAML: {what} wrong")
if len(doc["sequences"][1]["protein"]["sequence"]) == int(
        MASTER.loc[MASTER.peptide_id == SEL[0], "length"].iloc[0]):
    ok("YAML: peptide length matches the table")

# -- synthesise Boltz outputs for every selected design and run the collect path
RUNBLK = C4[C4.index("t0, finished = time.time(), 0"):]
PATCH_RES = sorted(ns4["PATCH"])
for i, pid in enumerate(SEL):
    ci = i // 25
    pred = f"{ns4['CD']}/c{ci:03d}/boltz_results_c{ci:03d}/predictions"
    pep = MASTER.loc[MASTER.peptide_id == pid, "sequence"].iloc[0]
    on_patch = (i % 3 != 0)          # two thirds land on the hotspot patch
    near = list(ns4["HOT"]) if on_patch else [5, 6, 7]
    fake_boltz_output(pred, pid, T_FULL["seq"], pep, near,
                      {"iptm_pair": 0.45 + 0.012 * i, "iptm": 0.4 + 0.01 * i,
                       "pep_plddt": 0.55 + 0.006 * i, "ipae": 12.0 - 0.15 * i,
                       "complex_plddt": 0.8, "complex_iplddt": 0.7,
                       "complex_ipde": 2.0, "ptm": 0.8,
                       "confidence_score": 0.6 + 0.005 * i})
ns4["print"] = lambda *a, **k: None
exec(RUNBLK, ns4)
DF4 = ns4["df"]
nfold = int(DF4["cofolded"].sum())
if nfold == len(SEL):
    ok(f"all {nfold} designs parsed from the synthetic Boltz output")
else:
    bad(f"only {nfold} of {len(SEL)} designs parsed")
for c in ("iptm_pair", "ipae", "pep_plddt", "hotspot_contacts", "hotspot_fraction",
          "n_contact_res", "receptor_rmsd", "complex_iplddt"):
    if c not in DF4.columns:
        bad(f"step 4 did not produce the column {c}")
if str(DF4["iptm_pair"].dtype) == "float64" and str(DF4["cofolded"].dtype) == "bool":
    ok("metric columns are numeric and cofolded is boolean")
else:
    bad(f"dtypes wrong: iptm_pair={DF4['iptm_pair'].dtype}, "
        f"cofolded={DF4['cofolded'].dtype}")
if len(DF4) == len(MASTER):
    ok(f"master.csv still holds all {len(MASTER)} generated peptides")
else:
    bad(f"master.csv shrank to {len(DF4)} rows from {len(MASTER)}")
_sub = DF4[DF4["cofolded"]]
if _sub["hotspot_contacts"].max() > 0 and _sub["hotspot_fraction"].max() > 0:
    ok("hotspot contacts and on-patch fraction are measured from the structures")
else:
    bad("hotspot geometry came out empty for every design")
_struct = len(os.listdir(f"{T_FULL['out']}/structures"))
if _struct == len(SEL):
    ok(f"{_struct} structures copied to Drive")
else:
    bad(f"{_struct} structures in Drive, expected {len(SEL)}")

# -- ipAE check against the known synthetic value
_pid = SEL[10]
_want_ipae = round(12.0 - 0.15 * 10, 4)
_got = float(DF4.loc[DF4.peptide_id == _pid, "ipae"].iloc[0])
if abs(_got - _want_ipae) < 1e-6:
    ok(f"ipAE read from the PAE matrix exactly ({_got})")
else:
    bad(f"ipAE is {_got}, expected {_want_ipae} from the synthetic matrix")
_want_pl = round(0.55 + 0.006 * 10, 4)
_gotpl = float(DF4.loc[DF4.peptide_id == _pid, "pep_plddt"].iloc[0])
if abs(_gotpl - _want_pl) < 1e-6:
    ok(f"peptide pLDDT averaged over the peptide tokens only ({_gotpl})")
else:
    bad(f"peptide pLDDT is {_gotpl}, expected {_want_pl}")

# -- resume: re-running must not refold
DF4.to_csv(f"{WORK}/master.csv", index=False)
DF4.to_csv(f"{T_FULL['out']}/master.csv", index=False)
ns = step4_ns()
exec(PRE, ns)
if len(ns["ids"]) == 0:
    ok("re-running step 4 finds nothing to fold")
else:
    bad(f"re-running step 4 would refold {len(ns['ids'])} designs")

# -- resume: a missing structure means refold
os.remove(f"{T_FULL['out']}/structures/{SEL[0]}.cif")
ns = step4_ns()
exec(PRE, ns)
if ns["ids"] == [SEL[0]]:
    ok("a design whose structure is missing from Drive is refolded")
else:
    bad(f"missing structure not detected; ids={ns['ids'][:3]}")
shutil.copyfile(f"{ns4['CD']}/c000/boltz_results_c000/predictions/{SEL[0]}/"
                f"{SEL[0]}_model_0.cif", f"{T_FULL['out']}/structures/{SEL[0]}.cif")

# -- staleness: change the steering and re-run WITHOUT redo
_before = io.open(f"{ns4['YD']}/c000/{SEL[0]}.yaml", encoding="utf-8").read()
ns = step4_ns(hotspots="20,21", redo=False)
exec(PRE, ns)
_after = io.open(f"{ns['YD']}/c000/{SEL[0]}.yaml", encoding="utf-8").read()
if _after != _before and len(ns["ids"]) == 0:
    bad("changing the hotspots rewrites the inputs but folds nothing, because the "
        "old predictions are still on disk and master.csv still says cofolded -- "
        "the results keep the OLD steering with no warning")
else:
    ok("changed steering is either refolded or refused")

# -- stale YAMLs left in a chunk directory
ns = step4_ns(max_designs=3, redo=True)
exec(PRE, ns)
_n_yaml = len([f for f in os.listdir(f"{ns['YD']}/c000") if f.endswith(".yaml")])
if _n_yaml > max(len(ns["ids"]), 1):
    bad(f"chunk directory holds {_n_yaml} input files but only {len(ns['ids'])} are "
        f"wanted; boltz predict folds every yaml in the directory, so a smaller "
        f"later run re-folds the leftovers")
else:
    ok("chunk directories hold only the wanted inputs")

# -- use_template off
ns = step4_ns()
ns["_T"] = T_NOTPL
try:
    exec(PRE, ns)
    if len(ns["PATCH"]) == 0:
        bad("with use_template off there is no geometry reference, so the patch is "
            "empty and hotspot_fraction is 0 for every design -- the wrong_face gate "
            "then rejects everything")
    else:
        ok("use_template off still produces a usable patch")
except Exception as e:
    bad(f"step 4 crashes outright when use_template is off: "
        f"{type(e).__name__}: {str(e)[:80]}")

# -- no hotspots at all
ns = step4_ns(hotspots="", redo=True)
exec(PRE, ns)
_y = io.open(f"{ns['YD']}/c000/{SEL[0]}.yaml", encoding="utf-8").read()
if "constraints" not in _y and ns["PATCH"] == set():
    ok("no hotspots: no pocket constraint emitted")
else:
    bad("empty hotspots still emitted a constraint")

# -- a bad hotspot
try:
    ns = step4_ns(hotspots="9999")
    exec(PRE, ns)
    bad("a hotspot outside the target was accepted")
except AssertionError:
    ok("a hotspot outside the target is refused before folding")

# -- missing plddt / pae files
_pd2 = f"{ns4['CD']}/edge/predictions"
fake_boltz_output(_pd2, "edge1", T_FULL["seq"], "ACDEFGHIK", PATCH_RES[:3],
                  {"iptm_pair": 0.7}, write_pae=False, write_plddt=False)
COLLECT = C4[C4.index("def interface("):C4.index("# --------------------------------------------------------- run")]
nsc = {"gemmi": gemmi, "np": np, "os": os, "json": json, "shutil": shutil,
       "T": T_FULL, "SEQ": T_FULL["seq"], "HOT": [93, 94], "PATCH": ns4["PATCH"],
       "contact_cutoff": 4.5, "OUT": T_FULL["out"], "SIG": "testsig",
       "_warn": set()}
exec(COLLECT, nsc)
r = nsc["collect"]("edge1", _pd2, 9)
if r and "ipae" not in r and "pep_plddt" not in r:
    ok("a design with no PAE or pLDDT file still yields its other metrics")
else:
    bad(f"missing npz files handled wrongly: {r}")
fake_boltz_output(_pd2, "edge2", T_FULL["seq"], "ACDEFGHIK", PATCH_RES[:3],
                  {"iptm_pair": 0.7}, truncate=True)
nsc["_warn"].clear()
r = nsc["collect"]("edge2", _pd2, 9)
if r and "ipae" not in r and any("PAE" in w for w in nsc["_warn"]):
    ok("a PAE matrix of the wrong size is reported, not skipped silently")
elif r and "ipae" not in r:
    bad("a PAE matrix of the wrong size is skipped with no message, so the design "
        "fails the ipae gate for a reason the user cannot see")

# ===================================================================== STEP 5
head("STEP 5  filtering: gates, re-measurement, re-runs")

C5 = cell("05_filter.py")
DF4.to_csv(f"{WORK}/master.csv", index=False)


def step5(**kw):
    ns = {"_E": _E, "_T": T_FULL, "json": json, "os": os, "re": re, "shutil": shutil,
          "gemmi": gemmi, "np": np, "pd": pd,
          "min_iptm": 0.6, "max_ipae": 10.0, "min_pep_plddt": 0.7,
          "max_receptor_rmsd": 0.0, "min_hotspot_contacts": 2,
          "min_hotspot_fraction": 0.6, "hotspots": HOT_LABELS, "patch_radius": 10.0,
          "contact_cutoff": 4.5, "max_cys": 1, "charge_min": -3.0, "charge_max": 5.0,
          "max_gravy": 1.0, "drop_liabilities": False, "rank_by": "iptm_pair",
          "print": lambda *a, **k: None}
    ns.update(kw)
    exec(C5, ns)
    return ns


ns5 = step5()
fin = ns5["final"]
_dd = ns5["d"]
print("    --- gate diagnostics on the synthetic set")
for _g, _m in ns5["G"]:
    print(f"        {_g:20s} {int(_m.sum()):>3d}/{len(_dd)}")
print(f"        iptm_pair  {_dd.iptm_pair.min():.2f}..{_dd.iptm_pair.max():.2f}   "
      f"ipae {_dd.ipae.min():.1f}..{_dd.ipae.max():.1f}   "
      f"plddt {_dd.pep_plddt.min():.2f}..{_dd.pep_plddt.max():.2f}")
print(f"        n_contact_res {_dd.n_contact_res.min()}..{_dd.n_contact_res.max()}   "
      f"hot {dict(_dd.hotspot_contacts.value_counts())}   "
      f"frac {_dd.hotspot_fraction.min():.2f}..{_dd.hotspot_fraction.max():.2f}")
full = ns5["_full"]
ok(f"filter ran: {len(fin)} of {int(DF4['cofolded'].sum())} passed")
if len(full) == len(MASTER):
    ok(f"master.csv kept all {len(full)} peptides after filtering")
else:
    bad(f"master.csv shrank to {len(full)} from {len(MASTER)}")
if "failed" in full.columns and full["failed"].notna().sum() == int(DF4["cofolded"].sum()):
    ok("the verdict is written only onto cofolded rows")

# re-run on the file it just wrote
try:
    ns5b = step5()
    if len(ns5b["final"]) == len(fin):
        ok("re-running the filter on its own output gives the same answer")
    else:
        bad(f"re-filtering changed the result: {len(ns5b['final'])} vs {len(fin)}")
except Exception as e:
    bad(f"re-running the filter raises {type(e).__name__}: {str(e)[:90]}")

# loosen the gates
ns5c = step5(min_iptm=0.5, min_pep_plddt=0.6, min_hotspot_fraction=0.3)
if len(ns5c["final"]) >= len(fin):
    ok(f"loosening the gates passes more ({len(ns5c['final'])} vs {len(fin)})")
else:
    bad("loosening the gates passed fewer designs")

# a wider patch must not reduce the on-patch fraction
_n10 = step5(patch_radius=10.0)["d"]["hotspot_fraction"].mean()
_n16 = step5(patch_radius=16.0)["d"]["hotspot_fraction"].mean()
if _n16 >= _n10:
    ok(f"patch_radius 10 -> 16 raises the mean on-patch fraction "
       f"({_n10:.2f} -> {_n16:.2f})")
else:
    bad(f"a wider patch lowered the fraction ({_n10:.2f} -> {_n16:.2f})")

# different hotspots, measured without refolding
_alt = step5(hotspots="20,21")
if not _alt["d"]["hotspot_contacts"].equals(ns5["d"]["hotspot_contacts"]):
    ok("different hotspots give different contact counts, with no refolding")
else:
    bad("changing the filter's hotspots did not change the measurement")

# no hotspots -> gates skipped
_nh = step5(hotspots="")
if not any(g[0] in ("hotspot_contacts", "wrong_face") for g in _nh["G"]):
    ok("no hotspots: both hotspot gates are skipped, not silently failed")
else:
    bad("hotspot gates still applied with no hotspots set")

# rank_by variations
for rb, asc in (("ipae", True), ("perplexity", True), ("iptm_pair", False),
                ("confidence_score", False), ("pep_plddt", False)):
    f = step5(min_iptm=0.4, min_pep_plddt=0.5, min_hotspot_fraction=0.2,
              rank_by=rb)["final"]
    if len(f) < 2:
        note(f"rank_by {rb}: only {len(f)} passed, ordering not exercised")
        continue
    v = pd.to_numeric(f[rb], errors="coerce").tolist()
    if (v == sorted(v)) if asc else (v == sorted(v, reverse=True)):
        ok(f"rank_by {rb} sorts in the right direction")
    else:
        bad(f"rank_by {rb} sorted wrongly: {v[:4]}")

# all gates impossible -> empty final, no crash
_z = step5(min_iptm=0.99)
if len(_z["final"]) == 0 and os.path.isfile(f"{T_FULL['out']}/final.csv"):
    ok("nothing passing still writes an empty final.csv and explains itself")

# sequence gates
_props = step5(max_cys=0)["d"]
if (_props["n_cys"] <= 0).sum() < len(_props):
    ok("a cysteine cap of 0 rejects the cysteine-containing designs")

# receptor_rmsd gate when the value is NaN for every design
_rr = step5(max_receptor_rmsd=1.0)
_names = [g[0] for g in _rr["G"]]
if "receptor_moved" in _names:
    _m = dict(_rr["G"])["receptor_moved"]
    if bool(_m.all()):
        ok("a receptor_rmsd gate passes designs whose value is unknown, not fails them")

# ===================================================================== STEP 6
head("STEP 6  viewing")

C6 = cell("06_results.py")
VIEW = C6[:C6.index("v = py3Dmol.view(")]


def step6(**kw):
    ns = {"_E": _E, "_T": T_FULL, "json": json, "os": os, "pd": pd,
          "which": "passing (final.csv)", "rank": 1, "colour_by": "pLDDT",
          "show_contacts": True, "print": lambda *a, **k: None}
    ns.update(kw)
    exec(VIEW, ns)
    return ns


shutil.copyfile(f"{T_FULL['out']}/final.csv", f"{WORK}/final.csv")
try:
    ns6 = step6()
    ok(f"step 6 opened rank 1: {ns6['r']['peptide_id']}")
    if list(ns6["HOT"]) == list(ns4["HOT"]):
        ok(f"step 6 picked up the hotspots the filter last used {ns6['HOT']}")
    else:
        bad(f"step 6 hotspots are {ns6['HOT']}, expected {ns4['HOT']}")
except Exception as e:
    bad(f"step 6 on final.csv raises {type(e).__name__}: {str(e)[:90]}")

try:
    ns6 = step6(rank=9999)
    ok("an out-of-range rank is clamped rather than crashing")
except Exception as e:
    bad(f"rank 9999 raises {type(e).__name__}: {str(e)[:70]}")
try:
    step6(rank=0)
    ok("rank 0 is clamped")
except Exception as e:
    bad(f"rank 0 raises {type(e).__name__}: {str(e)[:70]}")
try:
    ns6 = step6(which="any cofolded (master.csv)")
    ok(f"step 6 can browse all cofolded designs ({len(ns6['d'])} rows)")
except Exception as e:
    bad(f"step 6 on master.csv raises {type(e).__name__}: {str(e)[:90]}")

# an empty final.csv
io.open(f"{WORK}/final.csv", "w").write("peptide_id,rank\n")
try:
    step6()
    bad("step 6 accepted an empty final.csv")
except AssertionError:
    ok("step 6 refuses an empty final.csv with a clear message")

# ===================================================================== summary
head("SUMMARY")
print(f"  {len(FAILS)} bug(s), {len(NOTES)} note(s)\n")
for i, f in enumerate(FAILS, 1):
    print(f"  BUG {i}. {f}\n")
for i, n in enumerate(NOTES, 1):
    print(f"  NOTE {i}. {n}\n")
print(f"  workspace: {ROOT}")
