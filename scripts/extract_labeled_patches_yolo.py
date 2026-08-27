"""
Estrae patch multi-scala CON LE LORO LABEL — a differenza di
auto_generate_patches.py (che genera patch grezze senza etichette, per
il modello embedding), questo script ritaglia anche i poligoni delle
scaglie che cadono dentro ogni patch, li ricalcola in coordinate LOCALI
alla patch, e scrive un file YOLO-seg .txt per ognuna — pronte per il
training YOLO diretto sulle patch.

Le scaglie parzialmente dentro una patch vengono RITAGLIATE ai bordi
della patch (non scartate ne' incluse per intero) usando shapely.

Uso:
  pip install shapely
  python3 scripts/extract_labeled_patches_yolo.py --individual prime --scales 224 400 --output-size 224
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

RAW_ROOT = Path("data/raw")
LABELS_ROOT = Path("data/labels")
OUTPUT_ROOT = Path("data/yolo_dataset")


def log(msg: str) -> None:
    print(msg, flush=True)


def point_in_polygon(x: float, y: float, poly: list) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def compute_zone_hull(scaglie_polys: list) -> list:
    """Convex hull di tutte le scaglie labellate — stessa logica gia'
    testata in auto_generate_patches.py, per delimitare dove estrarre patch."""
    points = [tuple(p) for poly in scaglie_polys for p in poly]
    points = sorted(set(points))
    if len(points) <= 2:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def compute_patch_centers(zone: list, patch_size: int, stride: int, img_w: int, img_h: int) -> list:
    xs = [p[0] for p in zone]
    ys = [p[1] for p in zone]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    centers = []
    y = min_y
    while y + patch_size <= max_y:
        x = min_x
        while x + patch_size <= max_x:
            cx, cy = x + patch_size / 2, y + patch_size / 2
            if point_in_polygon(cx, cy, zone):
                centers.append((cx, cy))
            x += stride
        y += stride

    if not centers:
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        centers.append((cx, cy))
    return centers


def clip_polygon_to_patch(poly_points: list, px1: float, py1: float, px2: float, py2: float):
    """Ritaglia un poligono scaglia ai bordi della patch, ritorna lista
    di poligoni RISULTANTI in coordinate LOCALI alla patch (0,0 = angolo
    in alto a sinistra della patch), oppure [] se non interseca affatto."""
    from shapely.geometry import Polygon, box
    from shapely.validation import make_valid

    try:
        poly = Polygon(poly_points)
        if not poly.is_valid:
            poly = make_valid(poly)
    except Exception:
        return []

    patch_box = box(px1, py1, px2, py2)
    if not poly.intersects(patch_box):
        return []

    clipped = poly.intersection(patch_box)
    if clipped.is_empty:
        return []

    results = []
    geoms = clipped.geoms if clipped.geom_type == "MultiPolygon" else [clipped]
    for g in geoms:
        if g.geom_type != "Polygon" or g.area < 4:  # scarta frammenti minuscoli (rumore di ritaglio)
            continue
        local_coords = [[x - px1, y - py1] for x, y in g.exterior.coords]
        results.append(local_coords)
    return results


def polygon_to_yolo_line(coords: list, patch_w: int, patch_h: int, class_id: int = 0) -> str:
    norm = []
    for x, y in coords:
        norm.append(f"{max(0.0, min(1.0, x / patch_w)):.6f}")
        norm.append(f"{max(0.0, min(1.0, y / patch_h)):.6f}")
    return f"{class_id} " + " ".join(norm)


def process_individual(individual: str, scales: list, stride: int, output_size: int,
                        val_frac: float, output_name: str) -> dict:
    labels_path = LABELS_ROOT / f"{individual}.json"
    if not labels_path.exists():
        log(f"ERRORE: {labels_path.resolve()} non trovato.")
        return {"n_patches": 0}

    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
    log(f"Immagini nel file label: {len(labels_data)}")

    dataset_dir = OUTPUT_ROOT / output_name
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)

    base_size = min(scales)
    all_patch_records = []  # (patch_img, yolo_lines) da distribuire poi in train/val

    n_images_processed = 0
    n_images_skipped = 0
    start_time = time.time()

    for img_idx, (image_name, entry) in enumerate(labels_data.items()):
        img_start = time.time()
        log(f"\n--- Immagine {img_idx + 1}/{len(labels_data)}: {image_name} ---")

        img_path = RAW_ROOT / individual / image_name
        if not img_path.exists():
            log(f"  [SKIP] immagine non trovata su disco: {img_path.resolve()}")
            n_images_skipped += 1
            continue

        scaglie_polys = [
            [[float(x), float(y)] for x, y in s["coords"]]
            for s in (entry.get("scales") or [])
            if isinstance(s, dict) and s.get("type") == "polygon" and len(s.get("coords", [])) >= 3
        ]
        log(f"  Scaglie labellate: {len(scaglie_polys)}")
        if len(scaglie_polys) < 3:
            log(f"  [SKIP] troppe poche scaglie per calcolare una zona sensata")
            n_images_skipped += 1
            continue

        orig = Image.open(img_path).convert("RGB")
        img_w, img_h = orig.size
        log(f"  Dimensione immagine: {img_w}x{img_h}px")

        zone = compute_zone_hull(scaglie_polys)
        centers = compute_patch_centers(zone, base_size, stride, img_w, img_h)
        log(f"  Zona (convex hull): {len(zone)} vertici — centri patch candidati: {len(centers)}")

        n_patches_this_image = 0
        for scale in scales:
            n_out_of_bounds = 0
            n_empty = 0
            n_valid_this_scale = 0

            for (cx, cy) in centers:
                px1, py1 = cx - scale / 2, cy - scale / 2
                px2, py2 = cx + scale / 2, cy + scale / 2
                if px1 < 0 or py1 < 0 or px2 > img_w or py2 > img_h:
                    n_out_of_bounds += 1
                    continue

                yolo_lines = []
                for poly in scaglie_polys:
                    clipped_list = clip_polygon_to_patch(poly, px1, py1, px2, py2)
                    for clipped in clipped_list:
                        rescale = output_size / scale
                        rescaled = [[x * rescale, y * rescale] for x, y in clipped]
                        yolo_lines.append(polygon_to_yolo_line(rescaled, output_size, output_size))

                if not yolo_lines:
                    n_empty += 1
                    continue

                patch_img = orig.crop((int(px1), int(py1), int(px2), int(py2))).resize((output_size, output_size), Image.LANCZOS)
                all_patch_records.append((patch_img, yolo_lines))
                n_valid_this_scale += 1

            n_patches_this_image += n_valid_this_scale
            log(f"    scala {scale}px: {n_valid_this_scale} patch valide "
                f"({n_out_of_bounds} fuori bordo, {n_empty} senza scaglie visibili)")

        img_elapsed = time.time() - img_start
        log(f"  Totale questa immagine: {n_patches_this_image} patch — {img_elapsed:.1f}s")
        n_images_processed += 1

    total_elapsed = time.time() - start_time
    log(f"\n{'='*60}")
    log(f"RIEPILOGO ESTRAZIONE — {individual}")
    log(f"{'='*60}")
    log(f"Immagini processate: {n_images_processed} (saltate: {n_images_skipped})")
    log(f"Patch totali generate: {len(all_patch_records)}")
    log(f"Tempo totale: {total_elapsed:.1f}s ({total_elapsed/max(1,n_images_processed):.1f}s/immagine in media)")

    random.shuffle(all_patch_records)
    n_val = max(1, int(len(all_patch_records) * val_frac)) if all_patch_records else 0
    log(f"Split: {len(all_patch_records) - n_val} train, {n_val} val")

    log(f"\nSalvataggio file su disco ({len(all_patch_records)} patch)...")
    save_start = time.time()
    train_count = 0
    val_count = 0
    for i, (patch_img, yolo_lines) in enumerate(all_patch_records):
        split = "val" if i < n_val else "train"
        img_out = dataset_dir / f"images/{split}/patch_{i:05d}.jpg"
        img_out.parent.mkdir(parents=True, exist_ok=True)
        patch_img.save(img_out, format="JPEG", quality=90)

        label_out = dataset_dir / f"labels/{split}/patch_{i:05d}.txt"
        label_out.parent.mkdir(parents=True, exist_ok=True)
        label_out.write_text("\n".join(yolo_lines), encoding="utf-8")

        if split == "val":
            val_count += 1
        else:
            train_count += 1

        if (i + 1) % 500 == 0 or (i + 1) == len(all_patch_records):
            log(f"  ...{i + 1}/{len(all_patch_records)} salvate (train: {train_count}, val: {val_count}) — {time.time() - save_start:.1f}s finora")

    yaml_content = f"""# Generato da extract_labeled_patches_yolo.py — individuo: {individual}
path: {dataset_dir.resolve()}
train: images/train
val: images/val

names:
  0: scaglia
"""
    (dataset_dir / "data.yaml").write_text(yaml_content, encoding="utf-8")

    return {"n_patches": len(all_patch_records), "dataset_dir": dataset_dir}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--individual", type=str, required=True)
    parser.add_argument("--scales", type=int, nargs="+", default=[224])
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--max-patches", type=int, default=None,
                         help="se impostato, tiene al massimo questo numero di patch (campionate a caso) — utile per bilanciare reali vs sintetiche")
    args = parser.parse_args()

    scales = sorted(args.scales)
    stride = max(1, int(scales[0] * (1 - args.overlap)))
    output_name = args.output_name or f"{args.individual}_patches_labeled"

    log(f"Individuo: {args.individual} — scale: {scales} — output_size: {args.output_size}")

    result = process_individual(args.individual, scales, stride, args.output_size, args.val_frac, output_name)

    if args.max_patches and result["n_patches"] > args.max_patches:
        log(f"\nRiduco da {result['n_patches']} a {args.max_patches} patch (campionamento casuale)...")
        dataset_dir = result["dataset_dir"]
        # raccolgo tutti i file immagine/label generati, ne tengo solo max_patches a caso
        all_files = []
        for split in ["train", "val"]:
            for img_p in (dataset_dir / f"images/{split}").glob("*.jpg"):
                label_p = dataset_dir / f"labels/{split}/{img_p.stem}.txt"
                all_files.append((img_p, label_p))

        random.shuffle(all_files)
        to_remove = all_files[args.max_patches:]
        for img_p, label_p in to_remove:
            img_p.unlink(missing_ok=True)
            label_p.unlink(missing_ok=True)
        log(f"Rimossi {len(to_remove)} file in eccesso, tenuti {args.max_patches}.")

    log(f"\n=== FATTO ===")
    log(f"Dataset YOLO (patch labellate) pronto in: data/yolo_dataset/{output_name}/")
    log(f"Allena con: python3 scripts/train_yolo.py --data data/yolo_dataset/{output_name}/data.yaml")


if __name__ == "__main__":
    main()