"""
Pipeline COMPLETA E AUTOMATICA, un solo comando: da immagini labellate
(reali + N sintetiche) a dataset YOLO pronto per il training.

ORDINE DELLE OPERAZIONI (importante):
  1. Split a livello di IMMAGINE INTERA in train/val/test, PRIMA di
     tagliare qualsiasi patch — per ogni sorgente separatamente. Cosi'
     nessuna immagine finisce contemporaneamente in due split diversi
     (eviterebbe data leakage: patch della STESSA foto sono troppo
     simili tra loro, mescolarle tra train e test renderebbe la
     valutazione poco onesta).
  2. SOLO ORA si tagliano le patch multi-scala (con label), separatamente
     per ogni split — le patch di un'immagine "test" restano nel test,
     mai mischiate con train/val.
  3. Bilancia il conteggio: le sintetiche vengono ridotte (per ogni
     split separatamente) allo stesso numero della reale in quello split.
  4. Scrive tutto in UNA sola cartella dataset finale, pronta per
     scripts/train_yolo.py.

Uso:
  python scripts/build_full_yolo_pipeline.py --real prime --synthetic pattern_synth pattern_synth2 --scales 128 256 384 512 640 768 896 1024 --output-size 224 --output-name dataset_finale

  # per lanciare subito anche il training, alla fine:
  python scripts/build_full_yolo_pipeline.py --real prime --synthetic pattern_synth pattern_synth2 --output-name dataset_finale --auto-train
"""

import argparse
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from extract_labeled_patches_yolo import (
    compute_zone_hull,
    compute_patch_centers,
    clip_polygon_to_patch,
    polygon_to_yolo_line,
)

RAW_ROOT = Path("data/raw")
LABELS_ROOT = Path("data/labels")
OUTPUT_ROOT = Path("data/yolo_dataset")


def log(msg: str) -> None:
    print(msg, flush=True)


def split_images_by_name(image_names: list, val_frac: float, test_frac: float) -> dict:
    """Divide i NOMI IMMAGINE (non le patch) in train/val/test — questo
    accade PRIMA di tagliare qualsiasi patch, cosi' ogni immagine intera
    finisce interamente in UNO split solo."""
    names = list(image_names)
    random.shuffle(names)
    n = len(names)
    n_test = max(1, int(n * test_frac)) if n > 0 else 0
    n_val = max(1, int(n * val_frac)) if n > 0 else 0

    test = names[:n_test]
    val = names[n_test:n_test + n_val]
    train = names[n_test + n_val:]
    return {"train": train, "val": val, "test": test}


def generate_patch_records_for_images(individual: str, image_subset: list, scales: list,
                                       stride: int, output_size: int) -> list:
    """Genera patch (con label) SOLO per le immagini in image_subset —
    non per tutte quelle dell'individuo. Riusa gli helper gia' testati
    in extract_labeled_patches_yolo.py."""
    labels_path = LABELS_ROOT / f"{individual}.json"
    if not labels_path.exists():
        return []

    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
    records = []

    for image_name in image_subset:
        entry = labels_data.get(image_name)
        if entry is None:
            continue

        img_path = RAW_ROOT / individual / image_name
        if not img_path.exists():
            continue

        scaglie_polys = [
            [[float(x), float(y)] for x, y in s["coords"]]
            for s in (entry.get("scales") or [])
            if isinstance(s, dict) and s.get("type") == "polygon" and len(s.get("coords", [])) >= 3
        ]
        if len(scaglie_polys) < 3:
            continue

        orig = Image.open(img_path).convert("RGB")
        img_w, img_h = orig.size

        zone = compute_zone_hull(scaglie_polys)
        base_size = min(scales)
        centers = compute_patch_centers(zone, base_size, stride, img_w, img_h)

        for (cx, cy) in centers:
            for scale in scales:
                px1, py1 = cx - scale / 2, cy - scale / 2
                px2, py2 = cx + scale / 2, cy + scale / 2
                if px1 < 0 or py1 < 0 or px2 > img_w or py2 > img_h:
                    continue

                yolo_lines = []
                for poly in scaglie_polys:
                    for clipped in clip_polygon_to_patch(poly, px1, py1, px2, py2):
                        rescale = output_size / scale
                        rescaled = [[x * rescale, y * rescale] for x, y in clipped]
                        yolo_lines.append(polygon_to_yolo_line(rescaled, output_size, output_size))

                if not yolo_lines:
                    continue

                patch_img = orig.crop((int(px1), int(py1), int(px2), int(py2))).resize((output_size, output_size), Image.LANCZOS)
                records.append((patch_img, yolo_lines, individual))

    return records


def process_source(individual: str, scales: list, stride: int, output_size: int,
                    val_frac: float, test_frac: float, per_source_root: Path, force: bool = False) -> Path:
    """Elabora UNA sorgente (reale o sintetica) fino in fondo: split
    immagini -> patch -> salvataggio su disco nella SUA PROPRIA cartella
    (data/yolo_dataset/_per_source/{individual}/images/{train,val,test}/...).

    Se quella cartella esiste GIA' (ha almeno un file dentro images/train,
    val e test), salta interamente l'estrazione — non rifa' analisi e
    taglio patch gia' fatti in una run precedente. Usa --force per
    ignorare la cache e rifare tutto da capo.

    Non tiene mai piu' di UNA immagine PIL in memoria alla volta durante
    l'estrazione (scrive subito su disco appena tagliata una patch),
    e durante la fase di combinazione si copiano file gia' pronti invece
    di riaprire/rigenerare immagini — poco uso di RAM in entrambi i casi."""
    source_dir = per_source_root / individual

    already_done = all(
        (source_dir / f"images/{split}").exists() and any((source_dir / f"images/{split}").iterdir())
        for split in ["train", "val", "test"]
    ) if source_dir.exists() else False

    if already_done and not force:
        log(f"  '{individual}': cartella gia' presente ({source_dir}), SALTO estrazione (usa --force per rifare).")
        return source_dir

    if source_dir.exists() and force:
        log(f"  '{individual}': --force attivo, rifaccio da zero (elimino {source_dir})...")
        shutil.rmtree(source_dir)

    for sub in ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]:
        (source_dir / sub).mkdir(parents=True, exist_ok=True)

    labels_path = LABELS_ROOT / f"{individual}.json"
    if not labels_path.exists():
        log(f"  ERRORE: {labels_path.resolve()} non trovato, salto '{individual}'.")
        return source_dir

    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
    image_names = list(labels_data.keys())
    splits = split_images_by_name(image_names, val_frac, test_frac)
    log(f"  '{individual}': {len(image_names)} immagini -> "
        f"{len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

    for split_name in ["train", "val", "test"]:
        image_subset = splits[split_name]
        t0 = time.time()

        # genera E SALVA SUBITO ogni patch — non accumula immagini in RAM,
        # una alla volta: crop -> resize -> salva su disco -> scarta
        n_saved = 0
        counter = 0
        labels_path_local = LABELS_ROOT / f"{individual}.json"
        labels_data_local = json.loads(labels_path_local.read_text(encoding="utf-8"))

        for image_name in image_subset:
            entry = labels_data_local.get(image_name)
            if entry is None:
                continue
            img_path = RAW_ROOT / individual / image_name
            if not img_path.exists():
                continue

            scaglie_polys = [
                [[float(x), float(y)] for x, y in s["coords"]]
                for s in (entry.get("scales") or [])
                if isinstance(s, dict) and s.get("type") == "polygon" and len(s.get("coords", [])) >= 3
            ]
            if len(scaglie_polys) < 3:
                continue

            orig = Image.open(img_path).convert("RGB")
            img_w, img_h = orig.size
            zone = compute_zone_hull(scaglie_polys)
            base_size = min(scales)
            centers = compute_patch_centers(zone, base_size, stride, img_w, img_h)

            for (cx, cy) in centers:
                for scale in scales:
                    px1, py1 = cx - scale / 2, cy - scale / 2
                    px2, py2 = cx + scale / 2, cy + scale / 2
                    if px1 < 0 or py1 < 0 or px2 > img_w or py2 > img_h:
                        continue

                    yolo_lines = []
                    for poly in scaglie_polys:
                        for clipped in clip_polygon_to_patch(poly, px1, py1, px2, py2):
                            rescale = output_size / scale
                            rescaled = [[x * rescale, y * rescale] for x, y in clipped]
                            yolo_lines.append(polygon_to_yolo_line(rescaled, output_size, output_size))

                    if not yolo_lines:
                        continue

                    patch_img = orig.crop((int(px1), int(py1), int(px2), int(py2))).resize((output_size, output_size), Image.LANCZOS)

                    img_out = source_dir / f"images/{split_name}/{individual}_{counter:06d}.jpg"
                    patch_img.save(img_out, format="JPEG", quality=90)
                    label_out = source_dir / f"labels/{split_name}/{individual}_{counter:06d}.txt"
                    label_out.write_text("\n".join(yolo_lines), encoding="utf-8")

                    del patch_img  # esplicito: non serve piu', libera subito
                    counter += 1
                    n_saved += 1

            orig.close()
            del orig

        log(f"    {split_name}: {n_saved} patch salvate ({time.time() - t0:.1f}s)")

    return source_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, required=True, help="nome individuo con dati REALI")
    parser.add_argument("--synthetic", type=str, nargs="+", required=True, help="uno o piu' nomi individuo SINTETICI")
    parser.add_argument("--scales", type=int, nargs="+", default=[128, 256, 384, 512, 640, 768, 896, 1024])
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--val-frac", type=float, default=0)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--output-name", type=str, required=True)
    parser.add_argument("--auto-train", action="store_true", help="lancia subito anche train_yolo.py alla fine")
    parser.add_argument("--epochs", type=int, default=100, help="usato solo se --auto-train")
    parser.add_argument("--force", action="store_true",
                         help="ignora le cartelle per-sorgente gia' presenti e rifa' tutta l'estrazione da zero")
    args = parser.parse_args()

    scales = sorted(args.scales)
    stride = max(1, int(scales[0] * (1 - args.overlap)))

    all_sources = [("reale", args.real)] + [("sintetico", s) for s in args.synthetic]

    log(f"Sorgenti: {len(all_sources)} ({args.real} reale, {len(args.synthetic)} sintetiche)")
    log(f"Scale: {scales} — output_size: {args.output_size}px")
    log(f"Split (a livello di IMMAGINE, prima delle patch): "
        f"train {1 - args.val_frac - args.test_frac:.0%} / val {args.val_frac:.0%} / test {args.test_frac:.0%}\n")

    start_time = time.time()
    per_source_root = OUTPUT_ROOT / "_per_source"

    # --- FASE 1: per ogni sorgente, estrai (o riusa se gia' presente) la
    # sua cartella dedicata su disco — split immagine, poi patch, salvate
    # subito su disco una alla volta (mai tutte in RAM insieme) ---
    log("--- FASE 1: estrazione per-sorgente (con cache: salta se gia' presente) ---")
    source_dirs = {}
    for kind, individual in all_sources:
        log(f"\n'{individual}' ({kind}):")
        source_dirs[individual] = process_source(
            individual, scales, stride, args.output_size,
            args.val_frac, args.test_frac, per_source_root, force=args.force,
        )

    # --- FASE 2: bilancia il conteggio (conta file su disco, non oggetti
    # in memoria) e combina copiando i file gia' pronti ---
    log("\n--- FASE 2: bilanciamento e unione (copia file, nessuna immagine riaperta in RAM) ---")
    dataset_dir = OUTPUT_ROOT / args.output_name
    for sub in ["images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"]:
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)

    totals = {}
    for split_name in ["train", "val", "test"]:
        real_files = sorted((source_dirs[args.real] / f"images/{split_name}").glob("*.jpg"))
        target_count = len(real_files)
        log(f"  split '{split_name}': target = {target_count} patch (da '{args.real}')")

        chosen_per_source = {args.real: real_files}
        for name in args.synthetic:
            synth_files = sorted((source_dirs[name] / f"images/{split_name}").glob("*.jpg"))
            if len(synth_files) > target_count:
                random.shuffle(synth_files)
                synth_files = synth_files[:target_count]
                log(f"    '{name}': ridotto a {target_count} (copia selettiva)")
            else:
                log(f"    '{name}': {len(synth_files)} (meno del target, tenute tutte)")
            chosen_per_source[name] = synth_files

        all_chosen = [(f, individual) for individual in chosen_per_source for f in chosen_per_source[individual]]
        random.shuffle(all_chosen)
        totals[split_name] = len(all_chosen)

        log(f"  Copio '{split_name}': {len(all_chosen)} file...")
        for i, (img_file, individual) in enumerate(all_chosen):
            label_file = img_file.parent.parent.parent / f"labels/{split_name}/{img_file.stem}.txt"

            dest_img = dataset_dir / f"images/{split_name}/{individual}_{i:06d}.jpg"
            dest_label = dataset_dir / f"labels/{split_name}/{individual}_{i:06d}.txt"
            shutil.copy(img_file, dest_img)
            if label_file.exists():
                shutil.copy(label_file, dest_label)

            if (i + 1) % 1000 == 0 or (i + 1) == len(all_chosen):
                log(f"    ...{i + 1}/{len(all_chosen)} copiate")

    yaml_content = f"""# Generato da build_full_yolo_pipeline.py
# Sorgenti: {args.real} (reale), {', '.join(args.synthetic)} (sintetiche)
# Split fatto a livello di IMMAGINE INTERA prima di tagliare le patch
path: {dataset_dir.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: scaglia
"""
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    total_elapsed = time.time() - start_time

    log(f"\n{'='*60}")
    log(f"FATTO — dataset completo pronto in {total_elapsed:.1f}s")
    log(f"{'='*60}")
    log(f"Dataset: {dataset_dir.resolve()}")
    log(f"Totale: {sum(totals.values())} patch (train {totals['train']}, val {totals['val']}, test {totals['test']})")
    log(f"Split fatto a livello di IMMAGINE — nessuna patch della stessa foto e' finita in due split diversi.")
    log(f"Config: {yaml_path.resolve()}")

    if args.auto_train:
        log(f"\n--auto-train attivo: lancio train_yolo.py...")
        cmd = [sys.executable, str(Path(__file__).parent / "train_yolo.py"),
               "--data", str(yaml_path), "--epochs", str(args.epochs)]
        subprocess.run(cmd)
    else:
        log(f"\nPer allenare:")
        log(f"  python scripts/train_yolo.py --data {yaml_path.resolve()} --epochs 100")


if __name__ == "__main__":
    main()