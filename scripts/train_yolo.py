"""
Allena YOLO11-seg (segmentazione di istanza, Ultralytics) su uno o PIU'
dataset gia' convertiti (prepare_yolo_dataset.py per le immagini vere,
generate_copypaste_dataset.py per i sintetici).

Con piu' --data, li COMBINA in un unico training (Ultralytics supporta
liste di cartelle per train/val nel data.yaml) — cosi' alleni su
immagini vere e sintetiche insieme in un solo comando.

Parte da un checkpoint PRETRAINED (yolo11n-seg.pt di default — il piu'
piccolo/veloce, adatto a dataset ridotti) e fa fine-tuning sulle
scaglie. Ultralytics gestisce da solo augmentation, batching, mosaic,
ecc. — non serve il codice custom scritto per gli altri modelli.

Uso:
  pip install ultralytics --break-system-packages

  # un solo dataset
  python3 scripts/train_yolo.py --data data/yolo_dataset/prime/data.yaml

  # PIU' dataset insieme (veri + sintetici)
  python3 scripts/train_yolo.py --data data/yolo_dataset/prime/data.yaml data/yolo_dataset/prime_copypaste/data.yaml

  # modello piu' grande/capace (piu' lento):
  python3 scripts/train_yolo.py --data ... --model yolo11s-seg.pt
"""

import argparse
import json
import yaml
from datetime import datetime
from pathlib import Path

LOGS_ROOT = Path("models/training_logs")
LOGS_ROOT.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(msg, flush=True)


def load_dataset_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_split_dir(dataset_yaml: dict, base_dir: Path, split_key: str) -> str | None:
    """Ricava il percorso ASSOLUTO della cartella immagini per lo split
    (train/val), gestendo sia 'path' relativo che assoluto nel yaml."""
    rel = dataset_yaml.get(split_key)
    if not rel:
        return None
    ds_path = dataset_yaml.get("path")
    root = Path(ds_path) if ds_path else base_dir
    if not root.is_absolute():
        root = base_dir / root
    full = (root / rel).resolve()
    return str(full)


def build_combined_yaml(data_paths: list[Path], output_path: Path) -> Path:
    """Fonde piu' data.yaml in uno solo, con train/val come LISTE di
    cartelle — Ultralytics supporta nativamente questo formato, allenavA
    su tutte le sorgenti insieme senza dover copiare/unire i file."""
    train_dirs = []
    val_dirs = []
    names = None

    for p in data_paths:
        ds = load_dataset_yaml(p)
        base_dir = p.parent

        train_dir = resolve_split_dir(ds, base_dir, "train")
        val_dir = resolve_split_dir(ds, base_dir, "val")
        if train_dir:
            train_dirs.append(train_dir)
        if val_dir:
            val_dirs.append(val_dir)

        if names is None:
            names = ds.get("names")

    combined = {
        "train": train_dirs,
        "val": val_dirs,
        "names": names or {0: "scaglia"},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(combined, sort_keys=False), encoding="utf-8")
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, nargs="+", required=True,
                         help="uno o piu' path a data.yaml — con piu' di uno, li combina in un unico training")
    parser.add_argument("--model", type=str, default="yolo11s-seg.pt",
                         choices=["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt"],
                         help="checkpoint pretrained di partenza — n=nano (piu' veloce) ... l=large (piu' capace, piu' lento)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024, help="risoluzione training (YOLO ridimensiona le immagini a questo lato)")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--cache", type=str, default="disk", choices=["ram", "disk", "none"],
                         help="cache immagini per velocizzare le epoche successive: 'ram' tiene tutto in memoria "
                              "(veloce ma puo' saturare la RAM con molte immagini), 'disk' usa file temporanei su "
                              "disco (default, non tocca la RAM), 'none' non fa cache (piu' lento ma zero overhead)")
    parser.add_argument("--output-name", type=str, default=None)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        log("ERRORE: manca il pacchetto. Installa con:")
        log("  pip install ultralytics --break-system-packages")
        return

    data_paths = [Path(d) for d in args.data]
    for p in data_paths:
        if not p.exists():
            log(f"ERRORE: {p.resolve()} non trovato.")
            log("Lancia prima: python3 scripts/prepare_yolo_dataset.py --individual NOME")
            log("(oppure generate_copypaste_dataset.py per i sintetici)")
            return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if len(data_paths) == 1:
        data_yaml = data_paths[0]
        output_name = args.output_name or data_yaml.parent.name
        log(f"Dataset: {data_yaml.resolve()}")
    else:
        names_used = "_".join(p.parent.name for p in data_paths)
        output_name = args.output_name or names_used
        combined_yaml_path = Path("data/yolo_dataset") / f"_combined_{run_id}" / "data.yaml"
        data_yaml = build_combined_yaml(data_paths, combined_yaml_path)
        log(f"Dataset COMBINATO da {len(data_paths)} sorgenti:")
        for p in data_paths:
            log(f"  - {p.resolve()}")
        log(f"Config combinato generato: {data_yaml.resolve()}")

    log(f"Modello base: {args.model}")
    log(f"Epoche: {args.epochs} — imgsz: {args.imgsz} — batch: {args.batch} — cache: {args.cache}")

    model = YOLO(args.model)

    cache_arg = False if args.cache == "none" else args.cache  # ultralytics vuole False, 'ram' o 'disk'

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        cache=cache_arg,
        project="models/yolo_runs",
        name=f"{output_name}_{run_id}",
        patience=20,  # early stopping se non migliora per 20 epoche
        verbose=True,
    )

    final_metrics = {}
    try:
        final_metrics = {k: float(v) for k, v in results.results_dict.items()}
    except Exception:
        pass

    run_log = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model_architecture": f"YOLO11-seg ({args.model}) — fine-tuning da pretrained",
        "dataset_yaml": [str(p.resolve()) for p in data_paths],
        "combined_sources": len(data_paths) > 1,
        "hyperparameters": {
            "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch, "base_checkpoint": args.model,
        },
        "final_metrics": final_metrics,
        "output_dir": str((Path("models/yolo_runs") / f"{output_name}_{run_id}").resolve()),
    }
    log_path = LOGS_ROOT / f"{run_id}_{output_name}_yolo.json"
    log_path.write_text(json.dumps(run_log, indent=2, ensure_ascii=False), encoding="utf-8")

    log(f"\n=== FATTO ===")
    log(f"Risultati/pesi salvati in: models/yolo_runs/{output_name}_{run_id}/")
    log(f"  weights/best.pt  <- il checkpoint migliore, usa questo per predire")
    log(f"  weights/last.pt  <- l'ultimo, di solito peggiore di best.pt")
    log(f"Log esperimento: {log_path.resolve()}")

    log(f"\n{'='*60}")
    log(f"METRICHE FINALI (checkpoint best.pt)")
    log(f"{'='*60}")
    if final_metrics:
        def get_metric(key_substr):
            for k, v in final_metrics.items():
                if key_substr in k:
                    return v
            return None

        box_p = get_metric("precision(B)")
        box_r = get_metric("recall(B)")
        box_map50 = get_metric("mAP50(B)")
        box_map5095 = get_metric("mAP50-95(B)")
        mask_p = get_metric("precision(M)")
        mask_r = get_metric("recall(M)")
        mask_map50 = get_metric("mAP50(M)")
        mask_map5095 = get_metric("mAP50-95(M)")

        log(f"BOX (rilevamento):")
        log(f"  Precisione: {box_p:.3f}" if box_p is not None else "  Precisione: n/d")
        log(f"  Recall:     {box_r:.3f}" if box_r is not None else "  Recall: n/d")
        log(f"  mAP50:      {box_map50:.3f}" if box_map50 is not None else "  mAP50: n/d")
        log(f"  mAP50-95:   {box_map5095:.3f}" if box_map5095 is not None else "  mAP50-95: n/d")
        log(f"MASCHERA (segmentazione):")
        log(f"  Precisione: {mask_p:.3f}" if mask_p is not None else "  Precisione: n/d")
        log(f"  Recall:     {mask_r:.3f}" if mask_r is not None else "  Recall: n/d")
        log(f"  mAP50:      {mask_map50:.3f}" if mask_map50 is not None else "  mAP50: n/d")
        log(f"  mAP50-95:   {mask_map5095:.3f}" if mask_map5095 is not None else "  mAP50-95: n/d")
    else:
        log("Metriche non disponibili (formato risultati Ultralytics non riconosciuto — controlla il log esperimento JSON).")
    log(f"{'='*60}")

    log(f"\nPer predire su immagini nuove:")
    log(f"  from ultralytics import YOLO")
    log(f"  model = YOLO('models/yolo_runs/{output_name}_{run_id}/weights/best.pt')")
    log(f"  results = model('path/immagine.jpg')")


if __name__ == "__main__":
    main()