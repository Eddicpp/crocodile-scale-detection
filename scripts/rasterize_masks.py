"""
Converte i poligoni scaglia labellati (data/labels/{individuo}.json) in
maschere binarie a piena risoluzione (bianco = dentro una scaglia,
nero = sfondo/fessura) — servono come ground truth per allenare il
modello di segmentazione.

Uso:
  python3 scripts/rasterize_masks.py
  python3 scripts/rasterize_masks.py --individual prime
"""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

RAW_ROOT = Path("data/raw")
LABELS_ROOT = Path("data/labels")
MASKS_ROOT = Path("data/masks")
MASKS_ROOT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def find_raw_image(individual: str, image_name: str) -> Path | None:
    p = RAW_ROOT / individual / image_name
    return p if p.exists() else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--individual", type=str, default=None, help="se omesso: tutti")
    args = parser.parse_args()

    label_files = sorted(LABELS_ROOT.glob("*.json"))
    if args.individual:
        label_files = [f for f in label_files if f.stem == args.individual]

    log(f"File label trovati: {[f.name for f in label_files]}")

    total_masks = 0

    for lf in label_files:
        individual = lf.stem
        try:
            labels_data = json.loads(lf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log(f"[ERRORE] {lf} non e' JSON valido, skip.")
            continue

        out_dir = MASKS_ROOT / individual
        out_dir.mkdir(parents=True, exist_ok=True)

        for image_name, entry in labels_data.items():
            scales = entry.get("scales", []) or []
            polygons = [
                [(float(x), float(y)) for x, y in s["coords"]]
                for s in scales
                if isinstance(s, dict) and s.get("type") == "polygon" and len(s.get("coords", [])) >= 3
            ]

            log(f"\n--- {individual} / {image_name} — {len(polygons)} scaglie labellate ---")

            if not polygons:
                log("Nessuna scaglia labellata, skip.")
                continue

            img_path = find_raw_image(individual, image_name)
            if img_path is None:
                log(f"[ERRORE] immagine non trovata: data/raw/{individual}/{image_name}")
                continue

            orig = Image.open(img_path)
            mask = Image.new("L", orig.size, 0)
            draw = ImageDraw.Draw(mask)
            for poly in polygons:
                draw.polygon(poly, fill=255)

            out_path = out_dir / f"{Path(image_name).stem}.png"
            mask.save(out_path)
            log(f"Maschera salvata: {out_path.resolve()} — size {mask.size}")

            n_white = sum(1 for px in mask.getdata() if px > 0)
            pct = 100 * n_white / (mask.width * mask.height)
            log(f"Copertura scaglie: {pct:.1f}% dei pixel")

            total_masks += 1

    log(f"\n=== TOTALE MASCHERE GENERATE: {total_masks} ===")


if __name__ == "__main__":
    main()
