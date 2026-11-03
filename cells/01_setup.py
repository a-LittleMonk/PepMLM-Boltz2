#@title 1 &middot; Set up everything { display-mode: "form" }
#@markdown Run once per session. Installs PepMLM and Boltz-2 and fetches every weight
#@markdown the later steps need, so **nothing downloads mid-run**. Safe to re-run:
#@markdown finished phases are skipped and partial downloads resume.
drive_folder = "PepMLM"  #@param {type:"string"}
#@markdown Keep the 4 GB of Boltz weights in Drive, so the next session copies them
#@markdown instead of downloading them again:
cache_weights_in_drive = False  #@param {type:"boolean"}
#@markdown Fetch the 2 GB affinity checkpoint too. It only works on small molecules,
#@markdown never on peptides, so leave this off unless you have another use for it:
fetch_affinity_weights = False  #@param {type:"boolean"}
boltz_version = "2.2.1"  #@param {type:"string"}

import json, os, shutil, subprocess, sys, tarfile, time, urllib.request

T0 = time.time()
WORK   = "/content/pepmlm"
ENV    = "/content/boltz_env"
BCACHE = "/content/boltz_cache"
HFHOME = "/content/hf_cache"
STAMP  = "/content/.pepmlm_env_ok"
PEPMLM_REPO = "ChatterjeeLab/PepMLM-650M"

MOL_URL  = "https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar"
CONF_URL = "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt"
AFF_URL  = "https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt"


def phase(msg):
    print("\n" + "=" * 70)
    print(f"  {msg}    [{(time.time() - T0) / 60:.1f} min in]")
    print("=" * 70, flush=True)


def sh(cmd, env=None, quiet=True, check=True):
    """Run a command, streaming its output unless quiet."""
    p = subprocess.run(cmd, shell=isinstance(cmd, str), env=env,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0 or not quiet:
        tail = "\n".join(p.stdout.splitlines()[-40:]) if p.stdout else ""
        print(tail, flush=True)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")
    return p.stdout or ""


# ----------------------------------------------------------------- downloads
def _remote_size(url):
    try:
        req = urllib.request.Request(url, method="HEAD",
                                     headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=40) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return 0


_barw = [0]


def _progress(label, done, total, elapsed, moved, final=False):
    pct = 100.0 * done / total if total else 0.0
    rate = moved / elapsed if elapsed > 0.5 else 0.0
    eta = (total - done) / rate if rate > 0 and total else 0.0
    fill = int(28 * pct / 100)
    bar = "=" * fill + "-" * (28 - fill)
    txt = (f"  {label[:22]:22s} [{bar}] {pct:5.1f}%  "
           f"{done / 1e6:,.0f}/{total / 1e6:,.0f} MB  {rate / 1e6:4.1f} MB/s")
    txt += f"  done in {elapsed:.0f}s" if final else (
        f"  ~{eta / 60:.0f} min left" if eta > 90 else f"  ~{eta:.0f}s left")
    _barw[0] = max(_barw[0], len(txt))
    sys.stdout.write("\r" + txt.ljust(_barw[0]) + ("\n" if final else ""))
    sys.stdout.flush()
    if final:
        _barw[0] = 0


def fetch(url, path, label):
    """Resumable download that leaves a correctly-sized file alone."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    want = _remote_size(url)
    have = os.path.getsize(path) if os.path.isfile(path) else 0
    if want and have == want:
        print(f"  have   {label[:22]:22s} ({have / 1e6:,.0f} MB)", flush=True)
        return
    if want and have > want:
        os.remove(path)          # longer than the server's copy: start over
        have = 0
    req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    with urllib.request.urlopen(req, timeout=60) as r:
        if have and r.status != 206:
            have = 0             # server ignored the range: restart cleanly
        total = want or (have + int(r.headers.get("Content-Length") or 0))
        t0, last, done = time.time(), 0.0, have
        with open(path, "ab" if have else "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if time.time() - last > 0.4:
                    last = time.time()
                    _progress(label, done, total, last - t0, done - have)
    _progress(label, done, total, max(time.time() - t0, 0.1), done - have, final=True)
    if want and os.path.getsize(path) != want:
        raise RuntimeError(f"{label}: got {os.path.getsize(path)} bytes, expected {want}")


def same_size(a, b):
    return (os.path.isfile(a) and os.path.isfile(b)
            and os.path.getsize(a) == os.path.getsize(b))


# ----------------------------------------------------------------- 0. GPU
phase("GPU")
try:
    _q = sh(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader"]).strip().splitlines()[0]
    GPU_NAME, _mem, _cc = [x.strip() for x in _q.split(",")]
    GPU_MEM = int(_mem.split()[0])
    GPU_CC = float(_cc)
except Exception as e:
    raise RuntimeError(
        "No GPU found. Runtime > Change runtime type > GPU, then re-run.") from e

HAS_KERNELS = GPU_CC >= 8.0
print(f"  {GPU_NAME}   {GPU_MEM / 1024:.0f} GB   compute {GPU_CC}")
if HAS_KERNELS:
    TIER, N_COFOLD = "fast", 300
    print("  Optimised kernels available. Cofolding will be quick.")
else:
    TIER, N_COFOLD = "slow", 60
    print("  Compute < 8.0, so Boltz runs without its fused kernels (--no_kernels).")
    print(f"  Expect roughly 2-4 min per design on this GPU.")
print(f"  Default number of designs to cofold: {N_COFOLD} (you can change it in step 4)")

# ----------------------------------------------------------------- 1. Drive
phase("Google Drive")
from google.colab import drive
drive.mount("/content/drive")
DRIVE = f"/content/drive/MyDrive/{drive_folder.strip().strip('/')}"
for d in (DRIVE, f"{DRIVE}/inputs", f"{DRIVE}/weights", f"{DRIVE}/receptors"):
    os.makedirs(d, exist_ok=True)
os.makedirs(WORK, exist_ok=True)
print(f"  outputs will go to   My Drive/{drive_folder}/<run_name>/")
print(f"  put your PDB in      My Drive/{drive_folder}/inputs/")

# ----------------------------------------------------------------- 2. env
if os.path.exists(STAMP):
    phase("Environment (already built, skipping)")
else:
    phase("Python 3.11 for Boltz-2  (~1 min)")
    # Boltz needs 3.10-3.12; Colab's own interpreter may be outside that range, and
    # an isolated env also keeps Boltz's torch away from the one PepMLM uses.
    sh("apt-get -qq update")
    sh("apt-get -qq install -y python3.11 python3.11-venv python3.11-dev")
    sh(f"python3.11 -m venv {ENV}")
    sh(f"{ENV}/bin/pip install -q --no-cache-dir -U pip wheel")

    phase(f"Boltz-{boltz_version}  (~8 min, ~5 GB)")
    extra = "[cuda]" if HAS_KERNELS else ""
    sh(f"{ENV}/bin/pip install -q --no-cache-dir 'boltz{extra}=={boltz_version}'",
       quiet=False)

    phase("PepMLM dependencies in Colab's own Python  (~1 min)")
    sh(f"{sys.executable} -m pip install -q -U transformers huggingface_hub gemmi py3Dmol")
    open(STAMP, "w").write("ok")

BENV = {**os.environ,
        "PATH": f"{ENV}/bin:" + os.environ.get("PATH", ""),
        "PYTHONPATH": "",                 # Colab exports /env/python; it leaks
        "BOLTZ_CACHE": BCACHE,
        "HF_HOME": HFHOME}
os.environ["HF_HOME"] = HFHOME

# ----------------------------------------------------------------- 3. weights
phase("Boltz-2 weights  (~4 GB)")
os.makedirs(BCACHE, exist_ok=True)
_dw = f"{DRIVE}/weights"
for name, url in (("mols.tar", MOL_URL), ("boltz2_conf.ckpt", CONF_URL)):
    local, cached = f"{BCACHE}/{name}", f"{_dw}/{name}"
    if cache_weights_in_drive and same_size(cached, local):
        print(f"  have   {name[:22]:22s} (local)")
        continue
    if cache_weights_in_drive and os.path.isfile(cached) \
            and os.path.getsize(cached) == _remote_size(url):
        print(f"  copying {name} from Drive ...", flush=True)
        shutil.copyfile(cached, local)
        continue
    fetch(url, local, name)
    if cache_weights_in_drive:
        print(f"  copying {name} to Drive ...", flush=True)
        shutil.copyfile(local, cached)

# The affinity head is small-molecule only, so it is never loaded by this notebook.
# Boltz downloads it on every run if the file is absent, so leave a placeholder.
_aff = f"{BCACHE}/boltz2_aff.ckpt"
if fetch_affinity_weights:
    fetch(AFF_URL, _aff, "boltz2_aff.ckpt")
elif not os.path.exists(_aff):
    open(_aff, "wb").close()
    print("  skipped boltz2_aff.ckpt (small-molecule only); placeholder written")

if not os.path.isdir(f"{BCACHE}/mols") or not os.listdir(f"{BCACHE}/mols"):
    print("  extracting mols.tar (~2 min) ...", flush=True)
    with tarfile.open(f"{BCACHE}/mols.tar", "r") as tar:
        tar.extractall(BCACHE)
print(f"  component library: {len(os.listdir(f'{BCACHE}/mols')):,} files")

# ----------------------------------------------------------------- 4. PepMLM
phase("PepMLM-650M  (~2.6 GB)")
#@markdown
from huggingface_hub import snapshot_download
_pep = snapshot_download(PEPMLM_REPO, cache_dir=f"{HFHOME}/hub")
print(f"  {PEPMLM_REPO} -> {_pep}")

# ----------------------------------------------------------------- 5. check
phase("Checking the install")
_chk = sh(f"{ENV}/bin/python -c "
          f"\"import torch,boltz;print(torch.__version__,torch.cuda.is_available())\"",
          env=BENV).strip()
print(f"  boltz env: torch {_chk}")
assert _chk.endswith("True"), (
    "Boltz's torch cannot see the GPU. Do not re-run this cell in place -- "
    "Runtime > Disconnect and delete runtime, then start again.")
sh(f"{ENV}/bin/boltz predict --help", env=BENV)
print("  boltz CLI ok")

import torch
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(PEPMLM_REPO, cache_dir=f"{HFHOME}/hub")
print(f"  colab env: torch {torch.__version__}  cuda {torch.cuda.is_available()}")
assert torch.cuda.is_available(), "Colab's torch cannot see the GPU."

json.dump({"work": WORK, "env": ENV, "cache": BCACHE, "hf": HFHOME,
           "drive": DRIVE, "drive_folder": drive_folder, "repo": PEPMLM_REPO,
           "gpu": GPU_NAME, "gpu_mem": GPU_MEM, "gpu_cc": GPU_CC,
           "kernels": HAS_KERNELS, "tier": TIER, "n_cofold": N_COFOLD},
          open(f"{WORK}/env.json", "w"), indent=2)

phase(f"Ready in {(time.time() - T0) / 60:.1f} min -- go to step 2")
