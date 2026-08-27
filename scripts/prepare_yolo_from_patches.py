"""
Converte una cartella di patch gia' estratte in un dataset YOLO-seg pronto
per `scripts/train_yolo.py`.

Input atteso:
    data/patches/{run_folder}/{image_stem}/patch_0000_s256.png

Se c'e' `_meta.json` lo usa direttamente. Se manca, prova a ricostruire
centri e label leggendo `data/patch_zones/*__{image_stem}.bmp.json` e
`data/labels/{individual}.json`.

Uso:
  python3 scripts/prepare_yolo_from_patches.py --run-folder prime_dense128
  python3 scripts/prepare_yolo_from_patches.py --run-folder prime_dense128 --output-name prime_dense128_yolo --val-frac 0.15 --test-frac 0.0
"""

import argparse
import json
import math
import random
import shutil
from pathlib import Path

from PIL import Image

PATCHES_ROOT = Path("data/patches")
YOLO_ROOT = Path("data/yolo_dataset")
ZONES_ROOT = Path("data/patch_zones")
LABELS_ROOT = Path("data/labels")

DEFAULT_OVERLAP = 0.25


def log(msg: str) -> None:
    print(msg, flush=True)


def parse_patch_filename(patch_path: Path) -> tuple[int, int] | None:
    """patch_0007_s256.png -> (7, 256)."""
    stem = patch_path.stem
    if "_s" not in stem:
        return None
    try:
        idx_part, scale_part = stem.rsplit("_s", 1)
        idx = int(idx_part.split("_")[-1])
        scale = int(scale_part)
        return idx, scale
    except (ValueError, IndexError):
        return None


def load_meta(meta_path: Path) -> dict | None:
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def load_zone_for_stem(image_stem: str) -> tuple[str, str, list[list[float]]] | None:
    """Ritorna (individual, image_name, polygon) da data/patch_zones."""
    candidates = sorted(ZONES_ROOT.glob(f"*__{image_stem}*.json"))
    for zone_path in candidates:
        try:
            zone_data = json.loads(zone_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        zones = zone_data.get("zones", []) or []
        if not zones:
            continue
        polygon = zones[0]
        if not polygon or len(polygon) < 3:
            continue
        stem = zone_path.stem
        if "__" not in stem:
            continue
        individual, image_name = stem.split("__", 1)
        return individual, image_name, [[float(x), float(y)] for x, y in polygon]
    return None


def load_labels(individual: str) -> dict:
    labels_path = LABELS_ROOT / f"{individual}.json"
    if not labels_path.exists():
        return {}
    try:
        return json.loads(labels_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def compute_center_grid(polygon: list[list[float]], base_size: int, stride: int) -> list[tuple[float, float]]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    def point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
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

    centers = []
    if (max_x - min_x) >= base_size and (max_y - min_y) >= base_size:
        y = min_y
        while y + base_size <= max_y:
            x = min_x
            while x + base_size <= max_x:
                cx, cy = x + base_size / 2, y + base_size / 2
                if point_in_polygon(cx, cy, polygon):
                    centers.append((cx, cy))
                x += stride
            y += stride

    if not centers:
        centers.append((sum(xs) / len(xs), sum(ys) / len(ys)))

    return centers


def polygon_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    acc = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        acc += x1 * y2 - x2 * y1
    return abs(acc) * 0.5


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def transform_polygon_to_patch(poly: list[list[float]], box_x1: float, box_y1: float,
                               scale: int, out_w: int, out_h: int) -> list[list[float]] | None:
    ratio_x = out_w / scale
    ratio_y = out_h / scale

    transformed = [
        [clip((x - box_x1) * ratio_x, 0.0, float(out_w)), clip((y - box_y1) * ratio_y, 0.0, float(out_h))]
        for x, y in poly
    ]

    unique_points = []
    for x, y in transformed:
        pt = (round(x, 3), round(y, 3))
        if pt not in unique_points:
            unique_points.append(pt)

    if len(unique_points) < 3:
        return None

    pts = [(x, y) for x, y in transformed]
    if polygon_area(pts) < 1.0:
        return None

    return transformed


def polygon_to_yolo_line(coords: list[list[float]], img_w: int, img_h: int, class_id: int = 0) -> str:
    norm = []
    for x, y in coords:
        norm.append(f"{clip(x / img_w, 0.0, 1.0):.6f}")
        norm.append(f"{clip(y / img_h, 0.0, 1.0):.6f}")
    return f"{class_id} " + " ".join(norm)


def collect_patch_stems(run_dir: Path) -> list[Path]:
    stems = []
    for stem_dir in sorted(run_dir.iterdir()):
        if stem_dir.is_dir() and list(stem_dir.glob("patch_*.png")):
            stems.append(stem_dir)
    return stems


def split_stems(stems: list[Path], val_frac: float, test_frac: float, seed: int) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    ordered = stems[:]
    rng.shuffle(ordered)

    n_total = len(ordered)
    n_test = int(round(n_total * test_frac)) if test_frac > 0 else 0
    if test_frac > 0 and n_test == 0 and n_total >= 3:
        n_test = 1
    n_test = min(max(0, n_test), max(0, n_total - 1))

    test_stems = ordered[-n_test:] if n_test > 0 else []
    trainval = ordered[:-n_test] if n_test > 0 else ordered

    if len(trainval) >= 2:
        n_val = max(1, int(round(len(trainval) * val_frac))) if val_frac > 0 else 1
        n_val = min(n_val, len(trainval) - 1)
        val_stems = trainval[-n_val:]
        train_stems = trainval[:-n_val]
    else:
        train_stems = trainval
        val_stems = trainval

    return {"train": train_stems, "val": val_stems, "test": test_stems}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-folder", type=str, required=True, help="cartella in data/patches/ da convertire")
    parser.add_argument("--output-name", type=str, default=None, help="nome dataset in data/yolo_dataset/")
    parser.add_argument("--val-frac", type=float, default=0.15, help="quota di immagini/stem per validation")
    parser.add_argument("--test-frac", type=float, default=0.0, help="quota per test opzionale")
    parser.add_argument("--seed", type=int, default=42, help="seed per split riproducibile")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP,
                        help="overlap usato per ricostruire i centri quando manca _meta.json (default 0.25)")
    args = parser.parse_args()

    if not (0.0 <= args.val_frac < 1.0):
        log("ERRORE: --val-frac deve essere tra 0.0 e <1.0")
        return
    if not (0.0 <= args.test_frac < 1.0):
        log("ERRORE: --test-frac deve essere tra 0.0 e <1.0")
        return
    if not (0.0 <= args.overlap < 1.0):
        log("ERRORE: --overlap deve essere tra 0.0 e <1.0")
        return

    run_dir = PATCHES_ROOT / args.run_folder
    if not run_dir.exists():
        log(f"ERRORE: cartella non trovata: {run_dir.resolve()}")
        return

    stems = collect_patch_stems(run_dir)
    if not stems:
        log(f"ERRORE: nessuna sottocartella patch trovata in {run_dir.resolve()}")
        return

    output_name = args.output_name or args.run_folder
    dataset_dir = YOLO_ROOT / output_name
    for sub in ["images/train", "images/val", "labels/train", "labels/val", "images/test", "labels/test"]:
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)

    splits = split_stems(stems, args.val_frac, args.test_frac, args.seed)
    log(
        f"Run folder: {run_dir.resolve()} — stems: {len(stems)} — split: "
        f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
    )

    total_images = {"train": 0, "val": 0, "test": 0}
    total_labels = {"train": 0, "val": 0, "test": 0}

    for split_name, stem_dirs in splits.items():
        for stem_dir in stem_dirs:
            meta = load_meta(stem_dir / "_meta.json")
            if meta:
                centers = meta.get("centers", []) or []
                scales = meta.get("scales", []) or []
                labeled_scaglie = meta.get("labeled_scaglie", []) or []
                image_name = meta.get("image") or stem_dir.name
                individual = meta.get("individual")
            else:
                inferred = load_zone_for_stem(stem_dir.name)
                if not inferred:
                    log(f"  [SKIP] non riesco a ricostruire i metadati per: {stem_dir.name}")
                    continue
                individual, image_name, polygon = inferred
                labels_data = load_labels(individual)
                entry = labels_data.get(image_name, {})
                labeled_scaglie = []
                for item in entry.get("scales", []) or []:
                    if isinstance(item, dict) and item.get("type") == "polygon" and len(item.get("coords", [])) >= 3:
                        labeled_scaglie.append({
                            "scaglia_idx": len(labeled_scaglie),
                            "label": item.get("label", "scaglia"),
                            "polygon": [[float(x), float(y)] for x, y in item["coords"]],
                        })

                patch_files = sorted(stem_dir.glob("patch_*.png"))
                patch_scales = []
                for patch_path in patch_files:
                    parsed = parse_patch_filename(patch_path)
                    if parsed:
                        patch_scales.append(parsed[1])
                if not patch_scales:
                    log(f"  [SKIP] nessuna scala patch leggibile per: {stem_dir.name}")
                    continue
                base_size = min(patch_scales)
                stride = max(1, int(base_size * (1 - args.overlap)))
                centers = compute_center_grid(polygon, base_size, stride)
                scales = sorted(set(patch_scales))

            if not centers:
                log(f"  [SKIP] nessun centro valido per: {stem_dir.name}")
                continue

            patch_files = sorted(stem_dir.glob("patch_*.png"))
            if not patch_files:
                continue

            for patch_path in patch_files:
                parsed = parse_patch_filename(patch_path)
                if not parsed:
                    continue
                idx, scale = parsed
                if idx >= len(centers):
                    continue

                cx, cy = centers[idx]
                with Image.open(patch_path) as patch_img:
                    out_w, out_h = patch_img.size

                box_x1 = cx - scale / 2
                box_y1 = cy - scale / 2

                lines = []
                for item in labeled_scaglie:
                    poly = item.get("polygon")
                    if not poly or len(poly) < 3:
                        continue

                    xs = [p[0] for p in poly]
                    ys = [p[1] for p in poly]
                    if max(xs) < box_x1 or min(xs) > box_x1 + scale or max(ys) < box_y1 or min(ys) > box_y1 + scale:
                        continue

                    transformed = transform_polygon_to_patch(poly, box_x1, box_y1, scale, out_w, out_h)
                    if transformed is None:
                        continue
                    lines.append(polygon_to_yolo_line(transformed, out_w, out_h))

                rel_name = patch_path.name
                dest_img = dataset_dir / f"images/{split_name}" / rel_name
                dest_label = dataset_dir / f"labels/{split_name}" / f"{patch_path.stem}.txt"
                shutil.copy2(patch_path, dest_img)
                dest_label.write_text("\n".join(lines), encoding="utf-8")

                total_images[split_name] += 1
                total_labels[split_name] += len(lines)

            log(f"  {split_name}: {stem_dir.name} -> {len(patch_files)} patch copiate")

    yaml_lines = [
        f"path: {dataset_dir.resolve()}",
        "train: images/train",
        "val: images/val",
    ]
    if total_images["test"] > 0:
        yaml_lines.append("test: images/test")
    yaml_lines.extend([
        "",
        "names:",
        "  0: scaglia",
        "",
    ])
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text("\n".join(yaml_lines), encoding="utf-8")

    log("\n=== FATTO ===")
    log(f"Dataset YOLO creato in: {dataset_dir.resolve()}")
    log(f"data.yaml: {yaml_path.resolve()}")
    log(
        f"Immagini copiate: train={total_images['train']}, val={total_images['val']}, test={total_images['test']}"
    )
    log(
        f"Label scritte: train={total_labels['train']}, val={total_labels['val']}, test={total_labels['test']}"
    )
    log(f"Training: python scripts\\train_yolo.py --data {yaml_path.resolve()} --model yolo11n-seg.pt --epochs 200 --imgsz 512 --batch 16")


if __name__ == "__main__":
    main()