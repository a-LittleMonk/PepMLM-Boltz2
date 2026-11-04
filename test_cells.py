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


def run_target(crop, chain="A"):
    ns = {"gemmi": gemmi, "src": SRC, "target_chain": chain, "crop": crop,
          "use_template": True, "WORK": TMP, "re": re, "os": os,
          "print": lambda *a, **k: None}
    exec(BLOCK2, ns)
    return ns


ns = run_target("")
assert [t[0] for t in ns["res"]] == LABELS, [t[0] for t in ns["res"]]
assert ns["SEQ"] == AA1, ns["SEQ"]
ok(f"all 10 residues read in file order, water and chain B ignored -> {ns['SEQ']}")
assert len(ns["gaps"]) == 1 and ns["gaps"][0][:2] == ("22", "41"), ns["gaps"]
ok("the 22 -> 41 gap is reported, the insertion code does not create a false gap")

ns = run_target("17-22")
assert [t[0] for t in ns["res"]] == ["17", "18", "19", "20", "20A", "21", "22"]
ok("crop 17-22 keeps 7 residues, the insertion code 20A among them")

ns = run_target("17-19,41-43")
assert ns["SEQ"] == "ACD" + "ILK"[:0] + "IKL"[:0] + AA1[7:10], ns["SEQ"]
ok(f"two-range crop 17-19,41-43 gives {ns['SEQ']}")

for bad, why in ((("500-600",), "a crop that selects nothing"),):
    try:
        run_target(*bad)
        raise AssertionError(f"should have refused {why}")
    except AssertionError as e:
        if "should have refused" in str(e):
            raise
    ok(f"refuses {why}")

try:
    run_target("", chain="Z")
    raise AssertionError("should have refused a missing chain")
except AssertionError as e:
    if "should have" in str(e):
        raise
ok("refuses a chain that is not in the file")

ns = run_target("")
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
# Boltz matches a template's chains by subchain (label_asym_id); gemmi would call
# this "Axp" and Boltz refuses anything that is not the name in the YAML.
_subs = [x.subchain_id() for x in _b[0].subchains()]
assert _subs == ["A"], f"template subchain is {_subs}, Boltz needs ['A']"
_emap = {sc: e.name for e in _b.entities for sc in e.subchains}
assert all(s in _emap for s in _subs), f"{_subs} not in entity records {_emap}"
ok("template subchain is 'A' and its entity record agrees, so Boltz accepts it")
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
nsF = {"pd": pd, "np": np, "re": re, "d": d, "T": {"hotspots_idx": [4, 5]}, "HOT": [4, 5],
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

# ============================================== 5. output parsing order
print("\n5. step 4's reading of Boltz's output")
_c4 = io.open(os.path.join(CELLS, "04_cofold.py"), encoding="utf-8").read()
assert _c4.index('"Failed to process" in s') < _c4.index("_PROG.search(s)"), \
    "the refusal check must come before the progress-bar check"
ok("a refusal appended to the end of a tqdm progress line is still counted")

# ============================================== 6. dtypes of the results table
print("\n6. the results table's column types")
_c4src = io.open(os.path.join(CELLS, "04_cofold.py"), encoding="utf-8").read()
_n4 = {"pd": pd, "np": np, "re": re}
exec(_c4src[_c4src.index("_PROG = re.compile"):_c4src.index("done_before =")], _n4)
_t = pd.DataFrame({"peptide_id": ["a", "b", "c"], "length": [12, 13, 14]})
for _pid, _v in (("a", 0.81), ("b", 0.42)):      # c is deliberately never cofolded
    for _k, _val in (("cofolded", True), ("iptm_pair", _v), ("ipae", 5.0)):
        if _k not in _t.columns:
            _t[_k] = pd.NA
        _t.loc[_t.peptide_id == _pid, _k] = _val
try:
    _t.nlargest(2, "iptm_pair")
    raise AssertionError("object dtype should not have been rankable")
except TypeError:
    pass
_t = _n4["tidy"](_t)
assert str(_t["iptm_pair"].dtype) == "float64", _t["iptm_pair"].dtype
assert str(_t["cofolded"].dtype) == "bool", _t["cofolded"].dtype
_top = _t[_t["cofolded"]].nlargest(2, "iptm_pair")
assert list(_top["peptide_id"]) == ["a", "b"], list(_top["peptide_id"])
assert int(_t["cofolded"].sum()) == 2
ok("metric columns become numeric, so ranking works and uncofolded rows stay out")

# ============================================== 7. hotspots set in the filter cell
print("\n7. changing hotspots in the filter, without refolding")
_T = {"labels": LABELS, "seq": AA1, "hotspots_idx": [2, 8],
      "template": ns["TPL"], "out": TMP}


def resolve(text):
    n = {"T": _T, "hotspots": text, "print": lambda *a, **k: None}
    exec(slice_of("05_filter.py", 'LAB = T["labels"]',
                  "# ------------------------------------------------------------ "
                  "re-measure"), n)
    return n["HOT"]


assert resolve("") == [], resolve("")
ok("an empty field means no hotspot gates, not step 2's leftovers")
assert resolve("19,20A,43") == [3, 5, 10], resolve("19,20A,43")
ok("an override maps your own numbering, insertion code included, to 3, 5 and 10")
try:
    resolve("999")
    raise AssertionError("should have refused a residue outside the target")
except AssertionError as e:
    if "should have" in str(e):
        raise
ok("refuses a residue the target does not contain")

os.makedirs(TMP + "/structures", exist_ok=True)
complex_cif(TMP + "/structures/p1.cif", [4, 5, 6])
nsM = {"gemmi": gemmi, "np": np, "os": os, "contact_cutoff": 4.5,
       "HOT": [4, 5], "PATCH": {3, 4, 5, 6}}
exec(slice_of("05_filter.py", "def measure(cif):",
              'for c in ("n_contact_res", "contact_residues"'), nsM)
r1 = nsM["measure"](TMP + "/structures/p1.cif")
assert r1["hotspot_contacts"] == 2 and r1["hotspot_fraction"] == 1.0, r1
nsM["HOT"], nsM["PATCH"] = [9, 10], {8, 9, 10}
r2 = nsM["measure"](TMP + "/structures/p1.cif")
assert r2["hotspot_contacts"] == 0 and r2["hotspot_fraction"] == 0.0, r2
ok("one structure scores 2/1.0 against one hotspot pair and 0/0.0 against another")
nsM["HOT"], nsM["PATCH"], nsM["contact_cutoff"] = [4, 5], {3, 4, 5, 6}, 1.0
assert nsM["measure"](TMP + "/structures/p1.cif")["n_contact_res"] == 0
ok("tightening the contact cutoff to 1 A removes every contact, as it must")

# ============================================== 8. master.csv keeps every peptide
print("\n8. master.csv is not trimmed by filtering")
_df = pd.DataFrame({"peptide_id": ["p0", "p1", "p2", "p3", "p4"],
                    "sequence": ["AAAA"] * 5,
                    "selected": [True, True, False, False, False],
                    "cofolded": [True, True, False, False, False],
                    "iptm_pair": [0.8, 0.4, None, None, None]})
_d = _df[_df["cofolded"].eq(True)].copy()
_d["failed"] = ["", "iptm"]
_d["pass_all"] = [True, False]
_d["iptm_pair"] = [0.81, 0.42]            # as though re-measured in the filter
_n8 = {"df": _df, "d": _d, "pd": pd, "final": _d, "OUT": TMP, "WORK": TMP}
exec(slice_of("05_filter.py", "_full = df.set_index", '_full.to_csv(f"{WORK}'), _n8)
_out = _n8["_full"]
assert len(_out) == 5, "master lost rows: " + str(len(_out))
ok("all 5 generated peptides survive, not just the 2 that were cofolded")
assert list(_out["failed"].fillna("")) == ["", "iptm", "", "", ""]
ok("the filter's new columns are merged onto the cofolded rows only")
assert _out.set_index("peptide_id").at["p0", "iptm_pair"] == 0.81
ok("re-measured values overwrite step 4's, and uncofolded rows stay empty")

# ============================================== 9. step 4's steering hotspots
print("\n9. step 4 resolves its own steering hotspots")
_c4 = io.open(os.path.join(CELLS, "04_cofold.py"), encoding="utf-8").read()
_blk = _c4[_c4.index("# Hotspots, in your own numbering"):
           _c4.index("# Everything that changes what a fold produces")]


def steer(text):
    n = {"T": {"labels": LABELS}, "SEQ": AA1, "hotspots": text,
         "print": lambda *a, **k: None}
    exec(_blk, n)
    return n["HOT"]


assert steer("") == [], steer("")
ok("empty means no pocket constraint is emitted at all")
assert steer("18, 41") == [2, 8], steer("18, 41")
ok("spaces are tolerated and 18, 41 become positions 2 and 8")
assert steer("20A") == [5], steer("20A")
ok("an insertion code resolves, same as in the filter")
try:
    steer("777")
    raise AssertionError("should have refused a residue outside the target")
except AssertionError as e:
    if "should have" in str(e):
        raise
ok("refuses a residue the target does not contain, before any folding starts")

# ============================================== 10. step 4's resume behaviour
print("\n10. resuming step 4 after a disconnect")
_src4 = io.open(os.path.join(CELLS, "04_cofold.py"), encoding="utf-8").read()

# a) designs already folded in an earlier session are skipped
_skip = _src4[_src4.index("# Skip anything already folded"):
              _src4.index("if max_designs and len(todo)")]
_rdir = TMP + "/resume"
os.makedirs(_rdir + "/structures", exist_ok=True)
_df = pd.DataFrame({"peptide_id": ["p0", "p1", "p2"],
                    "selected": [True, True, True],
                    "cofolded": [True, True, False],
                    "fold_sig": ["sig1", "sig1", ""],
                    "perplexity": [1.0, 2.0, 3.0]})
io.open(_rdir + "/structures/p0.cif", "w").write("x")   # p1 has metrics but no file
_n = {"df": _df, "todo": _df[_df["selected"]].copy(), "os": os, "redo": False,
      "pd": pd, "SIG": "sig1", "OUT": _rdir, "print": lambda *a, **k: None}
exec(_skip, _n)
assert list(_n["todo"]["peptide_id"]) == ["p1", "p2"], list(_n["todo"]["peptide_id"])
ok("p0 is skipped; p1 is refolded because its structure is missing from Drive")

_n2 = {"df": _df, "todo": _df[_df["selected"]].copy(), "os": os, "redo": False,
       "pd": pd, "SIG": "sig2", "OUT": _rdir, "print": lambda *a, **k: None}
exec(_skip, _n2)
assert list(_n2["todo"]["peptide_id"]) == ["p0", "p1", "p2"],     list(_n2["todo"]["peptide_id"])
ok("a changed fold signature refolds p0 instead of reusing its old structure")

_n = {"df": _df, "todo": _df[_df["selected"]].copy(), "os": os, "redo": True,
      "pd": pd, "SIG": "sig1", "OUT": _rdir, "print": lambda *a, **k: None}
exec(_skip, _n)
assert list(_n["todo"]["peptide_id"]) == ["p0", "p1", "p2"]
ok("redo folds everything again, ignoring what is already saved")

# b) a half-written prediction folder is cleared, not skipped for ever
import textwrap
_part = textwrap.dedent(
    "    " + _src4[_src4.index("_part = [p for p in ch"):
                   _src4.index("have = sum(")])
_pred = TMP + "/pred"
for _p, _complete in (("good", True), ("cut_off", False)):
    os.makedirs(_pred + "/" + _p, exist_ok=True)
    if _complete:
        io.open(_pred + "/" + _p + "/confidence_" + _p + "_model_0.json", "w").write("{}")
_n = {"ch": ["good", "cut_off", "never_started"], "pred": _pred, "os": os,
      "shutil": __import__("shutil"), "ci": 0, "print": lambda *a, **k: None}
exec(_part, _n)
assert _n["_part"] == ["cut_off"], _n["_part"]
assert os.path.isdir(_pred + "/good"), "a finished design must be kept"
assert not os.path.isdir(_pred + "/cut_off"), "the half-written folder must be removed"
ok("an interrupted design's empty folder is deleted, so Boltz retries it")
ok("a finished design keeps its folder and is skipped")

# ============================================== 11. recovering the table from Drive
print("")
print("11. a fresh session recovers master.csv from Drive")
_rc = io.open(os.path.join(CELLS, "04_cofold.py"), encoding="utf-8").read()
_blk = _rc[_rc.index("def recover(name):"):_rc.index('assert os.path.isfile(recover(')]
_w, _o = TMP + "/fresh_work", TMP + "/fresh_drive"
os.makedirs(_o, exist_ok=True)
io.open(_o + "/master.csv", "w").write("peptide_id,selected" + chr(10) + "p0,True" + chr(10))
_nr = {"os": os, "shutil": __import__("shutil"), "WORK": _w, "OUT": _o,
       "E": {"drive_folder": "PepMLM"}, "T": {"run": "demo"},
       "print": lambda *a, **k: None}
exec(_blk, _nr)
assert not os.path.isdir(_w), "nothing should exist before the call"
assert _nr["recover"]("master.csv") == _w + "/master.csv"
assert os.path.isfile(_w + "/master.csv"), "the Drive copy was not brought back"
ok("with /content empty, the Drive copy is restored, so step 3 can be skipped")

io.open(_w + "/master.csv", "w").write("peptide_id,selected" + chr(10) + "local,True" + chr(10))
_nr["recover"]("master.csv")
assert "local" in io.open(_w + "/master.csv").read(), "a local file must not be clobbered"
ok("an existing local table is left alone, never overwritten by the Drive copy")

assert not os.path.isfile(_nr["recover"]("final.csv")), "must not invent a missing file"
ok("a table that exists in neither place is simply reported missing")

# ============================================== 12. step 5 is re-runnable
print("")
print("12. re-running the filter on a table it already filtered")
_blk5 = io.open(os.path.join(CELLS, "05_filter.py"), encoding="utf-8").read()
_blk5 = _blk5[_blk5.index("KD = dict("):_blk5.index('print(f"{len(d)} cofolded')]
_base = {"peptide_id": ["a"], "sequence": ["AKDEFGHIKLMNPQR"], "iptm_pair": [0.8],
         "ipae": [5.0], "pep_plddt": [0.85], "hotspot_contacts": [3],
         "hotspot_fraction": [0.9]}
_stale = {"n_cys": [9], "net_charge": [99.0], "gravy": [9.9], "liabilities": ["x"],
          "n_liabilities": [1], "failed": ["stale"], "pass_all": [False]}


def filt(frame):
    n = {"pd": pd, "np": np, "re": re, "d": frame.copy(), "HOT": [4, 5],
         "min_iptm": 0.6, "max_ipae": 10.0, "min_pep_plddt": 0.7,
         "max_receptor_rmsd": 0.0, "min_hotspot_contacts": 2,
         "min_hotspot_fraction": 0.6, "max_cys": 1, "charge_min": -3.0,
         "charge_max": 5.0, "max_gravy": 1.0, "drop_liabilities": False,
         "print": lambda *a, **k: None}
    exec(_blk5, n)
    return n["d"]


_fresh = filt(pd.DataFrame(_base))
_again = filt(pd.DataFrame({**_base, **_stale}))
assert not _again.columns.duplicated().any(), \
    sorted(set(_again.columns[_again.columns.duplicated()]))
ok("a table that already holds n_cys, net_charge, gravy ... gains no duplicates")
assert _again.at[0, "net_charge"] == _fresh.at[0, "net_charge"] != 99.0
ok("stale sequence properties are recomputed, not kept")
assert _again.at[0, "failed"] == _fresh.at[0, "failed"] == ""
ok("a stale verdict is replaced by this run's")

print("\nall checks passed")
