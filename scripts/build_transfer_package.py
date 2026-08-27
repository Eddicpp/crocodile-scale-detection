"""
Prepara una cartella "transfer_package/" con tutto il necessario per
continuare il lavoro su un altro computer: script + foto originali +
label. NON include patch, embedding, modelli allenati, checkpoint SAM
(quelli si rigenerano/riscaricano sul computer di arrivo).

Uso:
  python3 scripts/build_transfer_package.py
"""

import shutil
from pathlib import Path

SOURCE_ROOT = Path(".")
DEST_ROOT = Path("transfer_package")

# cosa includere
INCLUDE_DIRS = [
    "scripts",
    "data/raw",
    "data/labels",
    "data/masks",       # rasterizzate dai label, leggere, comodo riaverle
    "data/patch_zones",  # zone calcolate (convex hull ecc), piccole, utili
]

# cosa escludere ESPLICITAMENTE anche se dentro una cartella inclusa
EXCLUDE_PATTERNS = [
    "__pycache__",
    ".DS_Store",
    "*.pyc",
]

# cosa NON includere per niente (elencate solo per chiarezza nel riepilogo finale)
EXCLUDED_DIRS_INFO = [
    "data/patches       (si rigenerano con extract_patches.py / auto_generate_patches.py)",
    "data/embeddings    (si rigenerano con build_embeddings.py)",
    "data/yolo_dataset  (si rigenera con prepare_yolo_dataset.py)",
    "data/test_images   (output di test, non serve portarli)",
    "data/*preview*     (anteprime, si rigenerano)",
    "models/*.pt *.pth  (checkpoint allenati/pretrained — troppo grossi, si riallenano o riscaricano)",
    "models/training_logs (facoltativo: piccoli, valuta se vuoi la storia — non inclusi di default)",
]


def log(msg: str) -> None:
    print(msg, flush=True)


def should_exclude(path: Path) -> bool:
    return any(part in ("__pycache__",) or path.name == ".DS_Store" or path.suffix == ".pyc" for part in path.parts)


def copy_tree_filtered(src: Path, dst: Path) -> tuple[int, int]:
    """Copia ricorsivamente src->dst escludendo i pattern indesiderati.
    Ritorna (n_file_copiati, bytes_totali)."""
    n_files = 0
    total_bytes = 0
    for item in src.rglob("*"):
        if item.is_dir():
            continue
        if should_exclude(item):
            continue
        rel = item.relative_to(src)
        dest_path = dst / rel
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, dest_path)
        n_files += 1
        total_bytes += item.stat().st_size
    return n_files, total_bytes


def human_size(n_bytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}TB"


def main():
    if DEST_ROOT.exists():
        log(f"Pulisco cartella esistente: {DEST_ROOT.resolve()}")
        shutil.rmtree(DEST_ROOT)
    DEST_ROOT.mkdir(parents=True)

    log(f"Preparo pacchetto di trasferimento in: {DEST_ROOT.resolve()}\n")

    summary = []
    for rel_dir in INCLUDE_DIRS:
        src = SOURCE_ROOT / rel_dir
        if not src.exists():
            log(f"[SKIP] {rel_dir} non esiste, salto.")
            continue

        dst = DEST_ROOT / rel_dir
        n_files, n_bytes = copy_tree_filtered(src, dst)
        log(f"{rel_dir}: {n_files} file, {human_size(n_bytes)}")
        summary.append((rel_dir, n_files, n_bytes))

    # requirements.txt: lo generiamo se non esiste gia' nella root
    req_src = SOURCE_ROOT / "requirements.txt"
    req_dst = DEST_ROOT / "requirements.txt"
    if req_src.exists():
        shutil.copy2(req_src, req_dst)
        log("requirements.txt: copiato da quello esistente")
    else:
        req_dst.write_text(
            "streamlit\n"
            "streamlit-drawable-canvas\n"
            "pillow\n"
            "numpy\n"
            "opencv-python\n"
            "torch\n"
            "torchvision\n"
            "segment-anything\n"
            "ultralytics\n",
            encoding="utf-8",
        )
        log("requirements.txt: generato (non esisteva uno gia' pronto)")

    total_files = sum(s[1] for s in summary)
    total_bytes = sum(s[2] for s in summary)

    log(f"\n{'='*60}")
    log(f"RIEPILOGO")
    log(f"{'='*60}")
    log(f"Totale: {total_files} file, {human_size(total_bytes)}")
    log(f"\nCartella pronta: {DEST_ROOT.resolve()}")
    log(f"\nNON incluso (si rigenera sul computer di arrivo):")
    for line in EXCLUDED_DIRS_INFO:
        log(f"  - {line}")

    log(f"\nSul computer di arrivo, dopo aver copiato transfer_package/:")
    log(f"  1. Setup ambiente (venv + pip install -r requirements.txt)")
    log(f"  2. Scaricare il checkpoint SAM:")
    log(f"     curl -L -o models/sam_vit_b.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
    log(f"  3. Rigenerare quello che serve: extract_patches.py, build_embeddings.py, prepare_yolo_dataset.py, ecc.")


if __name__ == "__main__":
    main()
