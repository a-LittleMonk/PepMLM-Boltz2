"""Exercise the parts of the notebook that do not need a GPU.

Covers: residue numbering / crop / hotspot mapping against an awkward PDB, the Boltz
YAML writer, the contact + hotspot-patch geometry, and the filter gates.
"""
import io, json, os, re, tempfile
import gemmi
import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.join(HERE, "cells")
TMP = tempfile.mkdtemp()
AA3 = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU"]
AA1 = "ACDEFGHIKL"
ok = lambda m: print(f"  ok  {m}")


def slice_of(name, start, end):
    src = io.open(os.path.join(CELLS, name), encoding="utf-8").read()
    a = src.index(start)
    b = src.index(end)
    return src[a:b]


# ------------------------------------------------------------------ fixtures
def synthetic_pdb(path):
    """Chain A with a gap, an insertion code and non-sequential numbering;
    chain B to be ignored; a water to be stripped."""
    st = gemmi.Structure()
    st.spacegroup_hm = "P 1"
    m = gemmi.Model("1")
    labels = [(17, " "), (18, " "), (19, " "), (20, " "), (20, "A"),
              (21, " "), (22, " "), (41, " "), (42, " "), (43, " ")]
    ca = gemmi.Chain("A")
    for i, (num, ic) in enumerate(labels):
        r = gemmi.Residue()
        r.name = AA3[i]
        r.seqid = gemmi.SeqId(num, ic)
        for nm, off in (("N", -1.2), ("CA", 0.0), ("C", 1.2), ("CB", 0.6)):
            a = gemmi.Atom()
            a.name = nm
            a.element = gemmi.Element("N" if nm == "N" else "C")
            a.pos = gemmi.Position(i * 3.8 + off, 0.0, 0.0)
            r.add_atom(a)
        ca.add_residue(r)
    m.add_chain(ca)
    cb = gemmi.Chain("B")
    for i in range(5):
        r = gemmi.Residue()
        r.name = "GLY"
        r.seqid = gemmi.SeqId(100 + i, " ")
        a = gemmi.Atom()
        a.name, a.element = "CA", gemmi.Element("C")
        a.pos = gemmi.Position(0, 40 + i * 3.8, 0)
        r.add_atom(a)
        cb.add_residue(r)
    m.add_chain(cb)
    cw = gemmi.Chain("W")
    r = gemmi.Residue()
    r.name = "HOH"
    r.seqid = gemmi.SeqId(1, " ")
    a = gemmi.Atom()
    a.name, a.element = "O", gemmi.Element("O")
    a.pos = gemmi.Position(5, 5, 5)
    r.add_atom(a)
    cw.add_residue(r)
    m.add_chain(cw)
    st.add_model(m)
    st.setup_entities()
    st.write_pdb(path)
    return [f"{n}{ic.strip()}" for n, ic in labels]


SRC = os.path.join(TMP, "toy.pdb")
LABELS = synthetic_pdb(SRC)

# ============================================== 1. numbering / crop / hotspots
print("\n1. target parsing, cropping and hotspot mapping")
BLOCK2 = slice_of("02_target.py", "st = gemmi.read_structure(src)",
                  "# ------------------------------------------------------------------ 6. MSA")


def run_target(crop, hotspots, chain="A"):
    ns = {"gemmi": gemmi, "src": SRC, "target_chain": chain, "crop": crop,
          "hotspots": hotspots, "use_template": True, "WORK": TMP,
          "print": lambda *a, **k: None}
    exec(BLOCK2, ns)
    return ns


ns = run_target("", "")
assert [t[0] for t in ns["res"]] == LABELS, [t[0] for t in ns["res"]]
assert ns["SEQ"] == AA1, ns["SEQ"]
ok(f"all 10 residues read in file order, water and chain B ignored -> {ns['SEQ']}")
assert len(ns["gaps"]) == 1 and ns["gaps"][0][:2] == ("22", "41"), ns["gaps"]
ok("the 22 -> 41 gap is reported, the insertion code does not create a false gap")

ns = run_target("", "18,41")
assert ns["HOT"] == [2, 8], ns["HOT"]
ok("hotspots 18 and 41 map to sequence positions 2 and 8")

ns = run_target("17-22", "20A")
assert [t[0] for t in ns["res"]] == ["17", "18", "19", "20", "20A", "21", "22"]
assert ns["HOT"] == [5], ns["HOT"]
ok("crop 17-22 keeps 7 residues (20A included) and 20A maps to position 5")

ns = run_target("17-19,41-43", "")
assert ns["SEQ"] == "ACD" + "ILK"[:0] + "IKL"[:0] + AA1[7:10], ns["SEQ"]
ok(f"two-range crop 17-19,41-43 gives {ns['SEQ']}")

for bad, why in ((("", "99"), "a residue that does not exist"),
                 (("500-600", ""), "a crop that selects nothing")):
    try:
        run_target(*bad)
        raise AssertionError(f"should have refused {why}")
    except AssertionError as e:
        if "should have refused" in str(e):
            raise
    ok(f"refuses {why}")

try:
    run_target("", "", chain="Z")
    raise AssertionError("should have refused a missing chain")
except AssertionError as e:
    if "should have" in str(e):
        raise
ok("refuses a chain that is not in the file")

ns = run_target("", "18,41")
tpl = gemmi.read_structure(ns["TPL"])
nums = [r.seqid.num for r in tpl[0]["A"]]
assert nums == list(range(1, 11)), nums
assert gemmi.one_letter_code([r.name for r in tpl[0]["A"]]).upper() == AA1
ok("template is written as chain A renumbered 1..10, sequence unchanged")

# Boltz reads the template through make_structure_from_block and aligns the chain
# against the entity's sequence record. gemmi's setup_entities leaves that record
# empty, which made Boltz refuse every design with an IndexError.
_b = gemmi.make_structure_from_block(gemmi.cif.read(ns["TPL"])[0])
_b.merge_chain_parts()
_b.remove_empty_chains()
_fs = [gemmi.Entity.first_mon(x) for x in _b.entities[0].full_sequence]
assert len(_fs) == 10, f"template sequence record holds {len(_fs)} residues, not 10"
_al = gemmi.align_sequence_to_polymer(_fs, [x for x in _b[0].subchains()][0],
                                      gemmi.PolymerType.PeptideL,
                                      gemmi.AlignmentScoring())
assert len(_al.match_string) == 10, _al.match_string
for j in range(len(_al.match_string)):
    _fs[j]          # this is the exact line that used to raise IndexError
ok("template carries its sequence record, so Boltz's alignment walks it cleanly")

# ============================================== 2. the Boltz YAML
print("\n2. the YAML handed to Boltz-2")
T = {"msa": "/content/pepmlm/receptor_msa.csv",
     "template": "/content/pepmlm/receptor.cif", "apo": ""}
ns4 = {"SEQ": AA1, "T": T, "HOT": [2, 8], "pocket_max_distance": 6.0,
       "pocket_force": False, "open": open}
exec(slice_of("04_cofold.py", "def write_yaml(", "ids = list("), ns4)
p = os.path.join(TMP, "d.yaml")
ns4["write_yaml"](p, "PEPTIDESEQ")
y = yaml.safe_load(io.open(p).read())
assert y["version"] == 1
assert [s["protein"]["id"] for s in y["sequences"]] == ["A", "B"]
assert y["sequences"][0]["protein"]["sequence"] == AA1
assert y["sequences"][0]["protein"]["msa"] == T["msa"]
assert y["sequences"][1]["protein"]["sequence"] == "PEPTIDESEQ"
assert y["sequences"][1]["protein"]["msa"] == "empty"
assert y["templates"] == [{"cif": T["template"], "chain_id": "A", "template_id": "A"}]
pk = y["constraints"][0]["pocket"]
assert pk["binder"] == "B" and pk["contacts"] == [["A", 2], ["A", 8]]
assert pk["max_distance"] == 6.0 and pk["force"] is False
ok("valid YAML: receptor keeps the cached MSA, peptide is 'empty', "
   "pocket contacts are 1-based sequence indices")

ns4["HOT"] = []
ns4["T"] = {**T, "template": ""}
ns4["write_yaml"](p, "AAA")
y = yaml.safe_load(io.open(p).read())
assert "constraints" not in y and "templates" not in y
ok("no hotspots and no template -> neither block is emitted")

# ============================================== 3. contact geometry
print("\n3. contacts, hotspot patch and the wrong-face fraction")


def complex_cif(path, peptide_at, offset=3.0):
    """Receptor residues 1..10 on a line; a 3-residue peptide parked beside
    the residues listed in peptide_at."""
    st = gemmi.Structure()
    st.spacegroup_hm = "P 1"
    m = gemmi.Model("1")
    a = gemmi.Chain("A")
    for i in range(10):
        r = gemmi.Residue()
        r.name, r.seqid = AA3[i], gemmi.SeqId(i + 1, " ")
        at = gemmi.Atom()
        at.name, at.element = "CA", gemmi.Element("C")
        at.pos = gemmi.Position(i * 8.0, 0, 0)
        r.add_atom(at)
        a.add_residue(r)
    m.add_chain(a)
    b = gemmi.Chain("B")
    for k, idx in enumerate(peptide_at):
        r = gemmi.Residue()
        r.name, r.seqid = "GLY", gemmi.SeqId(k + 1, " ")
        at = gemmi.Atom()
        at.name, at.element = "CA", gemmi.Element("C")
        at.pos = gemmi.Position((idx - 1) * 8.0, offset, 0)
        r.add_atom(at)
        b.add_residue(r)
    m.add_chain(b)
    st.add_model(m)
    st.setup_entities()
    st.make_mmcif_document().write_file(path)


nsI = {"gemmi": gemmi, "np": np, "os": os, "T": {"apo": ""},
       "contact_cutoff": 4.5, "HOT": [4, 5], "PATCH": {3, 4, 5, 6}}
exec(slice_of("04_cofold.py", "def interface(", "def collect("), nsI)
interface = nsI["interface"]

c = os.path.join(TMP, "right.cif")
complex_cif(c, [4, 5, 6])
r = interface(c, 3)
assert r["n_contact_res"] == 3 and r["contact_residues"] == "4;5;6", r
assert r["hotspot_contacts"] == 2, r
assert r["hotspot_fraction"] == 1.0, r
ok("peptide on residues 4,5,6: 2 hotspots hit, on-patch fraction 1.0")

c = os.path.join(TMP, "wrong.cif")
complex_cif(c, [1, 2, 9])
r = interface(c, 3)
assert r["hotspot_contacts"] == 0, r
assert r["hotspot_fraction"] == 0.0, r
ok("peptide on the far face: 0 hotspots, fraction 0.0 -- caught by the wrong_face gate")

c = os.path.join(TMP, "half.cif")
complex_cif(c, [4, 9])
r = interface(c, 2)
assert r["hotspot_contacts"] == 1 and r["hotspot_fraction"] == 0.5, r
ok("peptide straddling both: 1 hotspot, fraction 0.5")

c = os.path.join(TMP, "far.cif")
complex_cif(c, [5], offset=12.0)
r = interface(c, 1)
assert r["n_contact_res"] == 0 and r["hotspot_fraction"] == 0.0, r
ok("peptide 12 A away: no contacts at all")

# ============================================== 4. the filter gates
print("\n4. filter gates")
rows = [
    # id            iptm  ipae plddt hcon hfrac  sequence          expect
    ("good",        0.80, 5.0, 0.85, 3, 0.90, "AKDEFGHIKLMNPQR", ""),
    ("low_iptm",    0.40, 5.0, 0.85, 3, 0.90, "AKDEFGHIKLMNPQR", "iptm"),
    ("bad_ipae",    0.80, 22.0, 0.85, 3, 0.90, "AKDEFGHIKLMNPQR", "ipae"),
    ("floppy",      0.80, 5.0, 0.40, 3, 0.90, "AKDEFGHIKLMNPQR", "plddt"),
    ("misses_hot",  0.80, 5.0, 0.85, 0, 0.90, "AKDEFGHIKLMNPQR", "hotspot_contacts"),
    ("wrong_face",  0.80, 5.0, 0.85, 3, 0.20, "AKDEFGHIKLMNPQR", "wrong_face"),
    ("cysteines",   0.80, 5.0, 0.85, 3, 0.90, "ACCDEFGHIKLMNPQ", "cys"),
    ("too_greasy",  0.80, 5.0, 0.85, 3, 0.90, "IVLFIVLFIVLFIVL", "gravy"),
    ("too_charged", 0.80, 5.0, 0.85, 3, 0.90, "KKKKKKRRRRRRGGG", "charge"),
]
d = pd.DataFrame([{"peptide_id": r[0], "iptm_pair": r[1], "ipae": r[2],
                   "pep_plddt": r[3], "hotspot_contacts": r[4],
                   "hotspot_fraction": r[5], "sequence": r[6]} for r in rows])
nsF = {"pd": pd, "np": np, "re": re, "d": d, "T": {"hotspots_idx": [4, 5]},
       "min_iptm": 0.6, "max_ipae": 10.0, "min_pep_plddt": 0.7,
       "max_receptor_rmsd": 0.0, "min_hotspot_contacts": 2,
       "min_hotspot_fraction": 0.6, "max_cys": 1, "charge_min": -3.0,
       "charge_max": 5.0, "max_gravy": 1.0, "drop_liabilities": False,
       "print": lambda *a, **k: None}
exec(slice_of("05_filter.py", "KD = dict(", 'print(f"{len(d)} cofolded'), nsF)
out = nsF["d"]
for rid, _, _, _, _, _, _, expect in rows:
    got = out.loc[out.peptide_id == rid, "failed"].iloc[0]
    assert got == expect, f"{rid}: failed='{got}', expected '{expect}'"
ok("each of the 9 synthetic designs fails exactly the gate it was built to fail")
assert int(out["pass_all"].sum()) == 1
ok("only the clean one passes every gate")

g = out.set_index("peptide_id")
assert g.loc["cysteines", "n_cys"] == 2
assert g.loc["too_charged", "net_charge"] == 12.0
assert g.loc["too_greasy", "gravy"] > 3.0
ok("cysteine count, net charge and GRAVY computed correctly")

lia = pd.DataFrame({"peptide_id": ["x"], "iptm_pair": [0.8], "ipae": [5.0],
                    "pep_plddt": [0.9], "hotspot_contacts": [3],
                    "hotspot_fraction": [0.9], "sequence": ["QDPNGDGMAAAAPPP"]})
nsF2 = {**nsF, "d": lia, "drop_liabilities": True}
exec(slice_of("05_filter.py", "KD = dict(", 'print(f"{len(d)} cofolded'), nsF2)
flags = nsF2["d"].loc[0, "liabilities"]
for want in ["N-terminal Gln", "Asp-Pro", "Asn-Gly", "Asp-Gly", "Met",
             "4+ identical", "3+ Pro"]:
    assert want in flags, f"{want} not flagged in '{flags}'"
assert nsF2["d"].loc[0, "failed"] == "liabilities"
ok("all seven sequence liabilities are detected and can gate")

print("\nall checks passed")
