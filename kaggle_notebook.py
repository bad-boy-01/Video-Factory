"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              NovelFactory — Kaggle Notebook Runner                         ║
║                                                                              ║
║  Structure:                                                                  ║
║    Cell 1 — Environment setup & dependency installation                     ║
║    Cell 2 — Mount project from GitHub (or upload dataset)                   ║
║    Cell 3 — Smoke test (import validation)                                   ║
║    Cell 4 — Run planning pipeline  (--mode plan)                             ║
║    Cell 5 — Run diffusion pipeline (--mode render)                           ║
║    Cell 6 — Inspect outputs                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

PASTE EACH SECTION INTO A SEPARATE KAGGLE CELL.
All cells are idempotent — safe to re-run.
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 1 — Environment setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import subprocess, sys, os

# ── 1a. HuggingFace cache — MUST be a WRITABLE directory ─────────────────────
# /kaggle/input/ is READ-ONLY — setting HF_HOME there causes a crash when
# transformers tries to write its cache index file.
# We use /kaggle/working/hf_cache/ for HF metadata (writable, ~20 GB space).
# The actual model weights are loaded directly from /kaggle/input/ by the
# _find_kaggle_model() helper in local_llm.py using local_files_only=True.
#
# How to attach models (Kaggle sidebar → Data → + Add Data → Models tab):
#   • "Qwen/Qwen1.5-4B-Chat"                    → attaches to /kaggle/input/qwen1-5-4b-chat/
#   • "stabilityai/stable-diffusion-xl-base-1.0" → attaches to /kaggle/input/stable-diffusion-xl-base-1-0/
import os
os.makedirs("/kaggle/working/hf_cache", exist_ok=True)
os.environ["HF_HOME"]               = "/kaggle/working/hf_cache"
os.environ["TRANSFORMERS_CACHE"]    = "/kaggle/working/hf_cache"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/kaggle/working/hf_cache"
os.environ["DIFFUSERS_CACHE"]       = "/kaggle/working/hf_cache"
# Tell local_llm.py where the Kaggle model datasets are mounted:
os.environ["KAGGLE_LLM_INPUT"]      = "/kaggle/input/qwen1-5-4b-chat"
os.environ["KAGGLE_SDXL_INPUT"]     = "/kaggle/input/stable-diffusion-xl-base-1-0"

# ── 1b. Verify GPU ────────────────────────────────────────────────────────────
import torch
print(f"CUDA available : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"VRAM           : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ── 1b. Install missing deps (Kaggle already has torch/transformers/diffusers) ─
EXTRA_DEPS = [
    "bitsandbytes>=0.46.1",   # 4-bit quantisation for LLM
    "accelerate",              # device_map="auto" support
    "safetensors",             # fast checkpoint loading
    "sentencepiece",           # Qwen tokenizer
    "python-docx",             # .docx novel input
    "pydantic>=2.0",           # domain models
]

print("\\nInstalling extra dependencies...")
for dep in EXTRA_DEPS:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", dep],
        capture_output=True, text=True
    )
    status = "✓" if result.returncode == 0 else "✗ " + result.stderr.strip()
    print(f"  {dep:<40} {status}")

print("\\n✅ Environment ready.")
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 2 — Mount the project
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, sys

# ── PREREQUISITE: Attach model datasets BEFORE running this notebook ──────────
# In the Kaggle sidebar: "Data" → "+ Add Data" → search the Models tab
#
#   1. Qwen/Qwen1.5-4B-Chat
#      Kaggle slug: qwen1-5-4b-chat
#      Will appear at: /kaggle/input/qwen1-5-4b-chat/
#
#   2. stabilityai/stable-diffusion-xl-base-1.0
#      Kaggle slug: stable-diffusion-xl-base-1-0
#      Will appear at: /kaggle/input/stable-diffusion-xl-base-1-0/
#
# After attaching, the os.environ lines in Cell 1 point HuggingFace directly
# there — no download occurs, no RAM is consumed during load.
# ─────────────────────────────────────────────────────────────────────────────

# ── Option A: GitHub clone (if repo is public) ────────────────────────────────
# Uncomment and fill in your repo URL:
# REPO_URL = "https://github.com/YOUR_USERNAME/NovelFactory.git"
# !git clone --depth 1 {REPO_URL} /kaggle/working/NovelFactory
# os.chdir("/kaggle/working/NovelFactory")

# ── Option B: Kaggle Dataset (recommended for private repos) ──────────────────
# 1. Zip your project: zip -r NovelFactory.zip NovelFactory/
# 2. Upload as a Kaggle Dataset named "novel-factory-src"
# 3. Attach it to this notebook, then uncomment:
# import zipfile
# with zipfile.ZipFile("/kaggle/input/novel-factory-src/NovelFactory.zip", "r") as z:
#     z.extractall("/kaggle/working/")
# os.chdir("/kaggle/working/NovelFactory")

# ── Option C: Already uploaded / cloned manually ────────────────────────────
PROJECT_DIR = "/kaggle/working/NovelFactory"   # ← adjust if needed
os.chdir(PROJECT_DIR)
print(f"Working directory: {os.getcwd()}")

# Add project root to Python path so 'import core.*' works
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
print(f"sys.path[0]      : {sys.path[0]}")

# Quick sanity check
assert os.path.isfile("main.py"), "main.py not found — check PROJECT_DIR"
print("✅ Project mounted.")
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 3 — Smoke test  (catches stale imports BEFORE wasting GPU time)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import pkgutil, importlib

print("Running import smoke test across core.*...")
failures = []

for module_info in pkgutil.walk_packages(["core"], prefix="core."):
    try:
        importlib.import_module(module_info.name)
        print(f"  ✓  {module_info.name}")
    except Exception as e:
        failures.append((module_info.name, str(e)))
        print(f"  ✗  {module_info.name}  →  {e}")

if failures:
    print(f"\\n❌ {len(failures)} import(s) failed:")
    for name, err in failures:
        print(f"   • {name}: {err}")
    raise RuntimeError("Fix import errors before continuing.")
else:
    print(f"\\n✅ All core.* modules imported cleanly.")
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 4 — Planning pipeline  (LLM stages, no GPU diffusion)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, subprocess, sys
from pathlib import Path

# ── 4a. Write (or specify) the input novel ────────────────────────────────────
NOVEL_PATH = "sample.txt"

NOVEL_TEXT = '''
Chapter 1 — The Forest
Alice stood at the edge of the dark forest, her blue dress catching the moonlight
like a lantern. The trees whispered secrets she couldn't yet understand.

Chapter 2 — The Stranger
A figure emerged from the shadows — tall, cloaked, carrying a lantern of
their own. "You shouldn't be here," he said, "but I'm glad you are."

Chapter 3 — The Choice
She had two paths: the road back to town, safe and familiar, or the winding
trail into the unknown. She had always chosen safety. Tonight felt different.
'''

with open(NOVEL_PATH, "w", encoding="utf-8") as f:
    f.write(NOVEL_TEXT.strip())
print(f"Novel written: {NOVEL_PATH}  ({len(NOVEL_TEXT)} chars)")

# ── 4b. Run planning (StoryBible → SceneSplitter → ShotPlanner →
#                     CameraPlanner → PromptBuilder → Validator → Timeline) ───
# First run: Qwen 1.5-4B-Chat (~2-3 GB) will be downloaded and cached.
# Subsequent runs: instant load from /root/.cache/huggingface

result = subprocess.run(
    [sys.executable, "main.py",
     "--novel", NOVEL_PATH,
     "--mode",  "plan"],
    text=True
)

if result.returncode != 0:
    raise RuntimeError("Planning pipeline failed — see output above.")
print("\\n✅ Planning pipeline complete.")
print("Artifacts:", [f for f in os.listdir("workspace") if f.endswith(".json")])
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 5 — Diffusion pipeline  (GPU image rendering)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, subprocess, sys

assert os.path.exists("workspace/PromptManifest.json"), (
    "PromptManifest.json not found — run Cell 4 first."
)

# ── Rendering scope (comment out flags you don't need) ────────────────────────
# Render everything:
RENDER_FLAGS = []

# Render a single shot:
# RENDER_FLAGS = ["--render-shot", "1"]

# Render a range of shots:
# RENDER_FLAGS = ["--render-shots", "1-5"]

# Render an entire scene:
# RENDER_FLAGS = ["--render-scene", "1"]

result = subprocess.run(
    [sys.executable, "main.py",
     "--novel", "sample.txt",
     "--mode",  "render",
     *RENDER_FLAGS],
    text=True
)

if result.returncode != 0:
    raise RuntimeError("Diffusion pipeline failed — see output above.")
print("\\n✅ Diffusion pipeline complete.")
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CELL 6 — Inspect outputs
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import os, json
from pathlib import Path
from IPython.display import display, Image as IPImage

WORKSPACE = Path("workspace")

# ── 6a. List all output files ─────────────────────────────────────────────────
print("=== workspace/ contents ===")
for p in sorted(WORKSPACE.rglob("*")):
    size = f"{p.stat().st_size:>8} B" if p.is_file() else ""
    print(f"  {size}  {p.relative_to(WORKSPACE)}")

# ── 6b. Pretty-print key JSON artifacts (first 40 lines each) ────────────────
for artifact in ["StoryBible.json", "SceneManifest.json", "ShotManifest.json",
                  "PromptManifest.json", "Timeline.json", "compile_report.json"]:
    path = WORKSPACE / artifact
    if path.exists():
        print(f"\\n─── {artifact} ───")
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
        print("".join(lines[:40]))
        if len(lines) > 40:
            print(f"  ... ({len(lines) - 40} more lines, open file to see all)")

# ── 6c. Director's report ─────────────────────────────────────────────────────
report = WORKSPACE / "directors_report.txt"
if report.exists():
    print("\\n═══ DIRECTOR'S REPORT ═══")
    print(report.read_text(encoding="utf-8"))

# ── 6d. Display rendered images ───────────────────────────────────────────────
images = sorted(WORKSPACE.rglob("image.png"))
if images:
    print(f"\\nRendered {len(images)} image(s):")
    for img_path in images[:12]:
        print(f"  {img_path}")
        display(IPImage(filename=str(img_path), width=512))
else:
    print("\\nNo rendered images found yet — run Cell 5 to generate them.")

# ── 6e. VRAM snapshot ─────────────────────────────────────────────────────────
try:
    import torch
    if torch.cuda.is_available():
        alloc   = torch.cuda.memory_allocated() / 1e9
        reservd = torch.cuda.memory_reserved()  / 1e9
        total   = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\\nVRAM: {alloc:.2f} allocated / {reservd:.2f} reserved / {total:.1f} total (GB)")
except ImportError:
    pass
"""
