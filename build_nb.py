"""Assemble PepMLM_Boltz2_colab.ipynb from the files in cells/.

Each cell lives in its own file so nothing has to be quoted twice. Run:
    python build_nb.py
"""
import io
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
NOTEBOOK_VERSION = "1.3.1"
OUT = os.path.join(HERE, "PepMLM_Boltz2_colab.ipynb")

ORDER = [
    ("00_header.md", "markdown"),
    ("01_setup.py", "code"),
    ("02_target.py", "code"),
    ("03_binder.py", "code"),
    ("04_cofold.py", "code"),
    ("05_filter.py", "code"),
    ("06_results.py", "code"),
    ("07_footer.md", "markdown"),
]


def lines(text):
    """ipynb stores source as a list of lines, each keeping its newline."""
    out = text.splitlines(keepends=True)
    if out and out[-1].endswith("\n"):
        out[-1] = out[-1][:-1]
    return out


cells = []
for name, kind in ORDER:
    path = os.path.join(HERE, "cells", name)
    src = io.open(path, encoding="utf-8").read()
    if kind == "code":
        # A cell must compile as plain Python; Colab's #@param lines are comments.
        probe = re.sub(r"(?m)^(\s*)!.*$", r"\1pass", src)
        try:
            compile(probe, name, "exec")
        except SyntaxError as e:
            raise SystemExit(f"{name}:{e.lineno}: {e.msg}\n  {e.text}") from e
        cells.append({"cell_type": "code", "execution_count": None,
                      "metadata": {"cellView": "form"},
                      "outputs": [], "source": lines(src)})
    else:
        cells.append({"cell_type": "markdown", "metadata": {},
                      "source": lines(src)})

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4",
                  "name": "PepMLM_Boltz2_colab.ipynb",
                  "toc_visible": True},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "notebook_version": NOTEBOOK_VERSION,
    },
    "cells": cells,
}

io.open(OUT, "w", encoding="utf-8", newline="\n").write(
    json.dumps(nb, indent=1, ensure_ascii=False) + "\n")

n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"wrote {OUT}")
print(f"  v{NOTEBOOK_VERSION}  {len(cells)} cells ({n_code} code), "
      f"{os.path.getsize(OUT) / 1024:.0f} KB")
