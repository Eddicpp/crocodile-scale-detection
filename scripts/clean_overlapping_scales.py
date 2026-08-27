"""
Ripulisce le scaglie labellate rimuovendo sovrapposizioni tra poligoni
adiacenti, garantendo un margine di separazione — i bordi non si
toccano/sovrappongono mai dopo la pulizia.

Metodo: ogni scaglia viene rasterizzata, poi ERODIA (ristretta) di un
margine configurabile (default 3px). Eventuali sovrapposizioni residue
tra maschere gia' erose vengono tagliate esplicitamente. Il poligono
finale viene ricostruito dalla maschera pulita.

DI DEFAULT NON SCRIVE NULLA in data/labels/ — genera solo immagini di
anteprima (bordi ORIGINALI in rosso sottile, bordi PULITI in verde
spesso, sovrapposti alla foto) cosi' puoi controllare visivamente prima
di applicare per davvero. Usa --apply per scrivere sul serio (con
backup automatico .bak).

Uso:
  # solo anteprima, nessuna modifica ai dati (default, sicuro)
  python3 scripts/clean_overlapping_scales.py
  python3 scripts/clean_overlapping_scales.py --individual prime --margin 5

  # applica per davvero, sovrascrive data/labels/{individuo}.json (con backup)
  python3 scripts/clean_overlapping_scales.py --apply
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RAW_ROOT = Path("data/raw")
LABELS_ROOT = Path("data/labels")
PREVIEW_ROOT = Path("data/cleaning_preview")


def log(msg: str) -> None:
    print(msg, flush=True)


def find_raw_image(individual: str, image_name: str) -> Path | None:
    p = RAW_ROOT / individual / image_name
    return p if p.exists() else None


def rasterize_polygon(poly: list, width: int, height: int) -> np.ndarray:
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([(x, y) for x, y in poly], fill=255)
    return np.array(mask) > 0


def erode_mask(mask: np.ndarray, margin: int) -> np.ndarray:
    if margin <= 0:
        return mask
    kernel = margin * 2 + 1
    img = Image.fromarray((mask * 255).astype("uint8"))
    eroded = img.filter(ImageFilter.MinFilter(kernel))
    return np.array(eroded) > 127


def mask_to_polygon(mask: np.ndarray, n_bins: int = 48) -> list | None:
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return None
    cx, cy = xs.mean(), ys.mean()
    angles = np.arctan2(ys - cy, xs - cx)
    dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    bin_idx = ((angles + np.pi) / (2 * np.pi) * n_bins).astype(int) % n_bins
    poly = []
    for b in range(n_bins):
        sel = bin_idx == b
        if not sel.any():
            continue
        i = np.argmax(dists[sel])
        poly.append([float(xs[sel][i]), float(ys[sel][i])])
    return poly if len(poly) >= 3 else None


def clean_scales_for_image(scales: list, width: int, height: int, margin: int) -> tuple[list, dict]:
    stats = {
        "originali": len(scales),
        "erose_ok": 0,
        "scartate_troppo_piccole": 0,
        "coppie_sovrapposte_trovate": 0,
        "dettagli_overlap": [],
        "dettagli_aree": [],
    }

    polygons = [s["coords"] for s in scales]
    labels = [s.get("label", "scaglia") for s in scales]

    raw_masks = [rasterize_polygon(p, width, height) for p in polygons]
    areas_before = [m.sum() for m in raw_masks]

    masks = [erode_mask(m, margin) for m in raw_masks]
    areas_after_erosion = [m.sum() for m in masks]

    n = len(masks)
    for i in range(n):
        for j in range(i + 1, n):
            if masks[i].sum() == 0 or masks[j].sum() == 0:
                continue
            overlap = masks[i] & masks[j]
            n_overlap_px = int(overlap.sum())
            if n_overlap_px > 0:
                masks[i] = masks[i] & ~overlap
                masks[j] = masks[j] & ~overlap
                stats["coppie_sovrapposte_trovate"] += 1
                stats["dettagli_overlap"].append(
                    f"scaglia #{i+1} <-> #{j+1}: {n_overlap_px}px sovrapposti, tagliati da entrambe"
                )

    new_scales = []
    new_masks_final = []
    for idx, (mask, label) in enumerate(zip(masks, labels)):
        area_final = int(mask.sum())
        stats["dettagli_aree"].append(
            f"scaglia #{idx+1} ({label}): {areas_before[idx]}px -> {areas_after_erosion[idx]}px dopo erosione -> {area_final}px dopo taglio overlap"
        )

        if area_final < 15:
            stats["scartate_troppo_piccole"] += 1
            new_masks_final.append(None)
            continue
        poly = mask_to_polygon(mask)
        if poly is None:
            stats["scartate_troppo_piccole"] += 1
            new_masks_final.append(None)
            continue
        new_scales.append({
            "type": "polygon",
            "coords": [[round(x, 2), round(y, 2)] for x, y in poly],
            "label": label,
        })
        new_masks_final.append(mask)
        stats["erose_ok"] += 1

    return new_scales, stats, raw_masks, new_masks_final


def make_preview_image(img_path: Path, raw_masks: list, new_masks_final: list) -> Image.Image:
    """Contorni ORIGINALI in rosso sottile, contorni PULITI in verde spesso,
    sovrapposti alla foto originale."""
    base = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(base, "RGBA")

    def mask_outline(mask: np.ndarray) -> np.ndarray:
        eroded_1px = np.array(
            Image.fromarray((mask * 255).astype("uint8")).filter(ImageFilter.MinFilter(3))
        ) > 127
        return mask & ~eroded_1px

    for mask in raw_masks:
        if mask is None or mask.sum() == 0:
            continue
        outline = mask_outline(mask)
        ys, xs = np.where(outline)
        for x, y in zip(xs, ys):
            draw.point((int(x), int(y)), fill=(255, 0, 0, 200))

    for mask in new_masks_final:
        if mask is None or mask.sum() == 0:
            continue
        outline_img = Image.fromarray((mask * 255).astype("uint8")).filter(ImageFilter.MaxFilter(3))
        thick_outline = (np.array(outline_img) > 127) & ~mask
        ys, xs = np.where(thick_outline)
        for x, y in zip(xs, ys):
            draw.point((int(x), int(y)), fill=(0, 255, 0, 255))

    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--individual", type=str, default=None, help="se omesso: tutti")
    parser.add_argument("--margin", type=int, default=3, help="pixel di separazione garantiti tra scaglie")
    parser.add_argument("--apply", action="store_true", help="scrive per davvero su data/labels/ (default: solo anteprima)")
    args = parser.parse_args()

    label_files = sorted(LABELS_ROOT.glob("*.json"))
    if args.individual:
        label_files = [f for f in label_files if f.stem == args.individual]

    if not label_files:
        log("Nessun file label trovato.")
        return

    log(f"=== clean_overlapping_scales ===")
    log(f"File label da processare: {[f.name for f in label_files]}")
    log(f"Margine separazione: {args.margin}px")
    log(f"Modalita': {'APPLY (scrive su data/labels/)' if args.apply else 'SOLO ANTEPRIMA (nessuna scrittura su data/labels/)'}")
    if not args.apply:
        log(f"Immagini di anteprima salvate in: {PREVIEW_ROOT.resolve()}")
        log("Rosso sottile = bordi ORIGINALI · Verde spesso = bordi DOPO la pulizia")

    for lf in label_files:
        individual = lf.stem
        try:
            labels_data = json.loads(lf.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log(f"[ERRORE] {lf} non e' JSON valido, skip.")
            continue

        log(f"\n{'='*60}")
        log(f"INDIVIDUO: {individual}")
        log(f"{'='*60}")
        changed = False

        for image_name, entry in labels_data.items():
            scales = entry.get("scales", []) or []
            if not scales:
                log(f"\n--- {image_name}: nessuna scaglia labellata, skip ---")
                continue

            img_path = find_raw_image(individual, image_name)
            if img_path is None:
                log(f"\n--- {image_name}: [ERRORE] immagine originale non trovata, skip ---")
                continue

            with Image.open(img_path) as img:
                width, height = img.size

            log(f"\n--- {image_name} ({width}x{height}px) — {len(scales)} scaglie ---")

            new_scales, stats, raw_masks, new_masks_final = clean_scales_for_image(scales, width, height, args.margin)

            log(f"  Aree per scaglia:")
            for line in stats["dettagli_aree"]:
                log(f"    {line}")

            if stats["dettagli_overlap"]:
                log(f"  Sovrapposizioni trovate e tagliate:")
                for line in stats["dettagli_overlap"]:
                    log(f"    {line}")
            else:
                log(f"  Nessuna sovrapposizione residua dopo erosione.")

            log(
                f"  RISULTATO: {stats['originali']} originali -> {stats['erose_ok']} pulite "
                f"({stats['coppie_sovrapposte_trovate']} coppie sovrapposte tagliate, "
                f"{stats['scartate_troppo_piccole']} scartate perche' troppo piccole)"
            )

            if args.apply:
                entry["scales"] = new_scales
                changed = True
            else:
                out_dir = PREVIEW_ROOT / individual
                out_dir.mkdir(parents=True, exist_ok=True)
                preview = make_preview_image(img_path, raw_masks, new_masks_final)
                out_path = out_dir / f"{Path(image_name).stem}_cleaning_preview.png"
                preview.save(out_path)
                log(f"  Anteprima salvata: {out_path.resolve()}")

        if args.apply and changed:
            backup_path = lf.with_suffix(".json.bak")
            shutil.copy(lf, backup_path)
            lf.write_text(json.dumps(labels_data, indent=2, ensure_ascii=False), encoding="utf-8")
            log(f"\nSalvato: {lf} (backup originale in {backup_path.name})")

    log(f"\n{'='*60}")
    log("FATTO")
    if not args.apply:
        log(f"Nessuna modifica scritta su data/labels/. Controlla le anteprime in {PREVIEW_ROOT.resolve()}")
        log("Se il risultato ti convince, rilancia con --apply per applicare davvero.")


if __name__ == "__main__":
    main()