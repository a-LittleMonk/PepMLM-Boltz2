#@title 2 &middot; Target { display-mode: "form" }
#@markdown Where the structure comes from. For a Drive file, give just the filename as
#@markdown it sits in `My Drive/<drive_folder>/inputs/`.
target_from = "Google Drive file"  #@param ["Google Drive file", "PDB ID (fetch from RCSB)", "path on disk"]
target_path = "receptor.pdb"  #@param {type:"string"}
target_chain = "A"  #@param {type:"string"}
#@markdown Region to keep, **in your file's own residue numbering** — e.g. `20-140`,
#@markdown or several ranges `20-140,190-230`. Leave empty to use the whole chain.
crop = ""  #@param {type:"string"}
#@markdown Name for this run's output folder (empty = derived from the filename):
run_name = ""  #@param {type:"string"}
use_template = True  #@param {type:"boolean"}
show_structure = True  #@param {type:"boolean"}

import glob, hashlib, json, os, shutil, subprocess, sys, time
import gemmi

E = json.load(open("/content/pepmlm/env.json"))
WORK, DRIVE, ENV, BCACHE = E["work"], E["drive"], E["env"], E["cache"]
BENV = {**os.environ, "PATH": f"{ENV}/bin:" + os.environ.get("PATH", ""),
        "PYTHONPATH": "", "BOLTZ_CACHE": BCACHE, "HF_HOME": E["hf"]}
MAX_ESM = 1024          # ESM-2 position limit; receptor + peptide must fit

# ------------------------------------------------------------------ 1. locate
if target_from.startswith("PDB ID"):
    pid = target_path.strip().lower().replace(".cif", "").replace(".pdb", "")
    assert len(pid) == 4, f"'{target_path}' is not a 4-character PDB id"
    src = f"{WORK}/{pid}.cif"
    if not os.path.isfile(src):
        import urllib.request
        urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pid}.cif", src)
    print(f"fetched {pid.upper()} from RCSB")
elif target_from.startswith("Google Drive"):
    src = f"{DRIVE}/inputs/{target_path.strip().lstrip('/')}"
    assert os.path.isfile(src), (
        f"not found: My Drive/{E['drive_folder']}/inputs/{target_path}\n"
        f"  files there now: {sorted(os.listdir(f'{DRIVE}/inputs')) or '(none)'}")
else:
    src = target_path.strip()
    assert os.path.isfile(src), f"not found: {src}"

stem = os.path.splitext(os.path.basename(src))[0]
RUN = (run_name.strip() or f"{stem}_{target_chain.strip()}")
RUN = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in RUN)
OUT = f"{DRIVE}/{RUN}"
for d in (OUT, f"{OUT}/structures", f"{OUT}/passing"):
    os.makedirs(d, exist_ok=True)
print(f"run name: {RUN}\noutputs:  My Drive/{E['drive_folder']}/{RUN}/")

# ------------------------------------------------------------------ 2. parse
st = gemmi.read_structure(src)
st.setup_entities()
st.remove_alternative_conformations()
st.remove_hydrogens()
st.remove_ligands_and_waters()
assert len(st) > 0, "no models in the structure"
model = st[0]

chains = [c.name for c in model]
ch_name = target_chain.strip()
assert ch_name in chains, f"chain '{ch_name}' not in this file. Chains: {chains}"

res = []                 # (auth_label, one-letter, gemmi residue)
for r in model[ch_name]:
    info = gemmi.find_tabulated_residue(r.name)
    if not (info and info.is_amino_acid()):
        continue
    if not r.find_atom("CA", "*"):
        continue
    letter = (info.one_letter_code or "x").upper()
    res.append((f"{r.seqid.num}{r.seqid.icode.strip()}", letter, r))
assert res, f"chain {ch_name} has no amino-acid residues with a CA atom"
print(f"chain {ch_name}: {len(res)} residues, "
      f"numbered {res[0][0]} to {res[-1][0]}")


def _ranges(text):
    out = []
    for part in text.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part.lstrip("-"):
            a, b = part.split("-", 1)
            out.append((int(a), int(b)))
        else:
            out.append((int(part), int(part)))
    return out


# ------------------------------------------------------------------ 3. crop
if crop.strip():
    keep = _ranges(crop)
    sel = [t for t in res
           if any(a <= int("".join(ch for ch in t[0] if ch.isdigit() or ch == "-")) <= b
                  for a, b in keep)]
    assert sel, f"crop {crop} selected nothing from chain {ch_name}"
    print(f"crop {crop}: kept {len(sel)} of {len(res)} residues")
    res = sel

SEQ = "".join(t[1] for t in res)
if "X" in SEQ:
    print(f"note: {SEQ.count('X')} non-standard residue(s) became X")

# gaps: residues the file does not observe are silently joined together
gaps = []
for i in range(1, len(res)):
    try:
        a, b = int(res[i - 1][0]), int(res[i][0])
    except ValueError:
        continue
    if b - a > 1:
        gaps.append((res[i - 1][0], res[i][0], b - a - 1))
if gaps:
    print(f"\n!! {len(gaps)} gap(s) in the observed numbering. The sequence sent to "
          f"the models joins them as if continuous:")
    for a, b, n in gaps[:8]:
        print(f"     {a} -> {b}   ({n} residue(s) missing)")
    print("   If a gap sits in the region you care about, crop to a gap-free "
          "stretch instead.")

# ------------------------------------------------------------------ 5. template
tpl = gemmi.Structure()
tpl.name = "receptor"
tpl.spacegroup_hm = "P 1"
m = gemmi.Model("1")
c = gemmi.Chain("A")
for i, (_lab, _let, r) in enumerate(res, start=1):
    nr = gemmi.Residue()
    nr.name, nr.seqid = r.name, gemmi.SeqId(i, " ")
    for a in r:
        nr.add_atom(a)
    c.add_residue(nr)
m.add_chain(c)
tpl.add_model(m)
tpl.setup_entities()
# setup_entities leaves _entity_poly with no sequence, and Boltz aligns the chain
# against that sequence. Against an empty one it runs off the end of the list with
# "IndexError: list index out of range" and silently skips every single design.
for ent in tpl.entities:
    if ent.entity_type == gemmi.EntityType.Polymer:
        ent.full_sequence = [r.name for r in c]
# Boltz names a template's chains by their mmCIF subchain (label_asym_id), not by the
# chain name, and gemmi invents subchain labels like "Axp" -- which Boltz then rejects
# with "Template chain A ... is not one of the protein chains!". Make them match.
for ch in tpl[0]:
    for r in ch:
        r.subchain = ch.name
for ent in tpl.entities:
    ent.subchains = ["A"]
tpl.assign_label_seq_id()
TPL = f"{WORK}/receptor.cif"
tpl.make_mmcif_document().write_file(TPL)

# Read it back exactly as Boltz will, and run the same checks it runs.
_chk = gemmi.make_structure_from_block(gemmi.cif.read(TPL)[0])
_chk.merge_chain_parts()
_chk.remove_waters()
_chk.remove_empty_chains()
_subs = [x.subchain_id() for x in _chk[0].subchains()]
_emap = {sc: e for e in _chk.entities for sc in e.subchains}
assert "A" in _subs, (
    f"the template's chain is labelled {_subs} rather than ['A'], so Boltz would "
    f"refuse it. Report this with your input file.")
assert all(s in _emap for s in _subs), (
    f"the template's chains {_subs} are not all in its entity records {list(_emap)}")
_fs = [gemmi.Entity.first_mon(x) for x in _chk.entities[0].full_sequence]
assert len(_fs) == len(res), (
    f"the template kept only {len(_fs)} of {len(res)} residues in its sequence "
    f"record; Boltz would refuse every design. Report this with your input file.")
_al = gemmi.align_sequence_to_polymer(
    _fs, [x for x in _chk[0].subchains()][0], gemmi.PolymerType.PeptideL,
    gemmi.AlignmentScoring())
assert len(_al.match_string) <= len(_fs), (
    "the template's sequence and its coordinates disagree, so Boltz would refuse it")
print(f"\ntemplate written and verified, chain A renumbered 1..{len(res)}")

# ------------------------------------------------------------------ 6. MSA + apo
H = hashlib.md5(SEQ.encode()).hexdigest()[:12]
CACHE = f"{DRIVE}/receptors/{H}"
MSA, APO = f"{WORK}/receptor_msa.csv", f"{WORK}/receptor_apo.cif"
if os.path.isfile(f"{CACHE}/msa.csv"):
    shutil.copyfile(f"{CACHE}/msa.csv", MSA)
    if os.path.isfile(f"{CACHE}/apo.cif"):
        shutil.copyfile(f"{CACHE}/apo.cif", APO)
    print(f"reusing the cached MSA for this exact sequence ({H})")
else:
    print("\nbuilding the receptor MSA once, to be reused by every design.")
    print("this also predicts the free receptor, which the filter uses later.\n")
    d = f"{WORK}/rec"
    os.makedirs(d, exist_ok=True)
    open(f"{d}/receptor.yaml", "w").write(
        "version: 1\nsequences:\n  - protein:\n      id: A\n"
        f"      sequence: {SEQ}\n")
    cmd = [f"{ENV}/bin/boltz", "predict", f"{d}/receptor.yaml",
           "--out_dir", d, "--use_msa_server", "--cache", BCACHE,
           "--recycling_steps", "3", "--sampling_steps", "50",
           "--diffusion_samples", "1", "--output_format", "mmcif",
           "--num_workers", "2"]
    if not E["kernels"]:
        cmd.append("--no_kernels")
    t0 = time.time()
    p = subprocess.run(cmd, env=BENV, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True)
    hits = sorted(glob.glob(f"{d}/**/msa/*.csv", recursive=True))
    if not hits:
        print("\n".join(p.stdout.splitlines()[-40:]))
        raise RuntimeError("the MSA server did not return an alignment -- see above")
    shutil.copyfile(hits[0], MSA)
    apo = sorted(glob.glob(f"{d}/**/predictions/**/*_model_0.cif", recursive=True))
    if apo:
        shutil.copyfile(apo[0], APO)
    os.makedirs(CACHE, exist_ok=True)
    shutil.copyfile(MSA, f"{CACHE}/msa.csv")
    if apo:
        shutil.copyfile(APO, f"{CACHE}/apo.cif")
    open(f"{CACHE}/sequence.txt", "w").write(SEQ + "\n")
    print(f"  MSA + free receptor done in {(time.time() - t0) / 60:.1f} min, "
          f"cached as receptors/{H}")

_depth = sum(1 for _ in open(MSA)) - 1
print(f"MSA depth: {_depth:,} sequences")
if _depth < 100:
    print("!! a shallow MSA usually means a sequence the databases barely cover; "
          "Boltz will be less confident on this target")

# ------------------------------------------------------------------ 7. save
json.dump({"run": RUN, "out": OUT, "src": src, "chain": ch_name,
           "seq": SEQ, "n_res": len(res), "crop": crop.strip(),
           "labels": [t[0] for t in res],
           "template": TPL if use_template else "", "msa": MSA, "apo": APO,
           "hash": H, "max_peptide": MAX_ESM - len(SEQ)},
          open(f"{WORK}/target.json", "w"), indent=2)

print(f"\nreceptor is {len(SEQ)} residues, so the longest peptide PepMLM can take "
      f"is {MAX_ESM - len(SEQ)}")
assert len(SEQ) < MAX_ESM - 10, (
    f"the receptor is {len(SEQ)} residues and ESM-2 stops at {MAX_ESM}. "
    f"Crop it to the domain you care about.")

# ------------------------------------------------------------------ 8. view
if show_structure:
    import py3Dmol
    v = py3Dmol.view(width=760, height=460)
    v.addModel(open(TPL).read(), "cif")
    v.setStyle({"cartoon": {"color": "grey", "opacity": 0.85}})
    v.zoomTo()
    v.show()
    print("This is the target as steps 3 and 4 will see it. Hotspots are set in"
          " step 4 (to steer the folding) and step 5 (to judge the result).")
