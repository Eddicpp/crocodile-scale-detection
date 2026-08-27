"""
Valuta un modello YOLO attuale sul dataset sintetico generato.

Uso di esempio:
  ./gucci/bin/python scripts/eval_on_synthetic.py \
    --model models/yolo_runs/prime_20260823_.../weights/best.pt \
    --data data/yolo_dataset/prime_copypaste/data.yaml \
    --split val \
    --imgsz 1024
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="percorso al file best.pt da valutare")
    parser.add_argument("--data", type=str, required=True, help="percorso del data.yaml sintetico")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"], help="split da usare per la validazione")
    parser.add_argument("--imgsz", type=int, default=1024, help="risoluzione di validazione")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="IoU threshold")
    args = parser.parse_args()

    model_path = Path(args.model)
    data_yaml = Path(args.data)

    if not model_path.exists():
        raise FileNotFoundError(f"Modello non trovato: {model_path.resolve()}")
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset sintetico non trovato: {data_yaml.resolve()}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Manca ultralytics. Installa con: pip install ultralytics --break-system-packages") from exc

    model = YOLO(str(model_path))
    print(f"[eval] modello={model_path.resolve()}")
    print(f"[eval] dataset={data_yaml.resolve()} split={args.split}")
    print(f"[eval] imgsz={args.imgsz} conf={args.conf} iou={args.iou}")

    metrics = model.val(
        data=str(data_yaml),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=True,
    )

    print("\n=== RISULTATI VALUTAZIONE SINTETICA ===")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.p:.4f}")
    print(f"Recall: {metrics.box.r:.4f}")


if __name__ == "__main__":
    main()
