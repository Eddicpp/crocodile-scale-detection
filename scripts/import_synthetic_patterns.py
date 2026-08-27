"""
Importa le immagini sintetiche generate da un altro tool (cartella con
pattern_NNN_grid_WxH.png + pattern_NNN_grid_WxH_labels.json per ognuna)
nel formato standard del progetto: data/raw/{nome}/ + data/labels/{nome}.json

Il formato dei label sorgente e' GIA' identico a quello usato nel
progetto (stessa struttura: chiave=nome immagine, "scales" con poligoni
in pixel assoluti) — questo script si limita a copiare le immagini e
FONDERE tutti i singoli {pattern}_labels.json in un unico file combinato,
cosi' diventa immediatamente compatibile con auto_generate_patches.py,
prepare_yolo_dataset.py, ecc. senza scrivere nuovo codice di parsing.

Uso:
  python3 scripts/import_synthetic_patterns.py --source "data/sintetiche/prova 1/generated_patterns" --output-name pattern_synth
"""

import argparse
import json
import shutil
from pathlib import Path

RAW_ROOT = Path("data/raw")
LABELS_ROOT = Path("data/labels")
IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".bmp"]


def log(msg: str) -> None:
    print(msg, flush=True)


def find_image_for_stem(source_dir: Path, stem: str) -> Path | None:
    for ext in IMAGE_EXTENSIONS:
        candidate = source_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="cartella con le immagini + {nome}_labels.json")
    parser.add_argument("--output-name", type=str, required=True, help="nome individuo di destinazione (data/raw/{nome}/, data/labels/{nome}.json)")
    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.exists():
        log(f"ERRORE: cartella sorgente non trovata: {source_dir.resolve()}")
        return

    label_files = sorted(source_dir.glob("*_labels.json"))
    log(f"File label trovati: {len(label_files)}")

    out_raw_dir = RAW_ROOT / args.output_name
    out_raw_dir.mkdir(parents=True, exist_ok=True)

    combined_labels = {}
    n_images_copied = 0
    n_scales_total = 0
    n_skipped = 0

    for label_path in label_files:
        stem = label_path.stem[: -len("_labels")]  # "pattern_000_grid_15x20_labels" -> "pattern_000_grid_15x20"

        img_path = find_image_for_stem(source_dir, stem)
        if img_path is None:
            log(f"  [SKIP] {label_path.name}: nessuna immagine trovata per stem '{stem}'")
            n_skipped += 1
            continue

        try:
            data = json.loads(label_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            log(f"  [SKIP] {label_path.name}: JSON non valido ({e})")
            n_skipped += 1
            continue

        # il file sorgente ha gia' il nome immagine come chiave (stessa
        # struttura del progetto) — normalmente una sola chiave per file
        for image_key, entry in data.items():
            dest_img_name = img_path.name  # manteniamo il nome file originale
            dest_img_path = out_raw_dir / dest_img_name
            shutil.copy(img_path, dest_img_path)

            combined_labels[dest_img_name] = entry
            n_scales = len(entry.get("scales", []) or [])
            n_scales_total += n_scales
            n_images_copied += 1

    labels_out_path = LABELS_ROOT / f"{args.output_name}.json"
    labels_out_path.parent.mkdir(parents=True, exist_ok=True)
    labels_out_path.write_text(json.dumps(combined_labels, indent=2, ensure_ascii=False), encoding="utf-8")

    log(f"\n=== FATTO ===")
    log(f"Immagini copiate: {n_images_copied} (saltate: {n_skipped})")
    log(f"Scaglie totali importate: {n_scales_total}")
    log(f"Immagini in: {out_raw_dir.resolve()}")
    log(f"Label combinate in: {labels_out_path.resolve()}")
    log(f"\nOra compatibile con tutta la pipeline esistente, es:")
    log(f"  python3 scripts/auto_generate_patches.py --individual {args.output_name} --scales 224 400 --output-size 224")


if __name__ == "__main__":
    main()
