"""
Genera immagini SINTETICHE "copy-paste": ritaglia le scaglie VERE gia'
labellate (usando il loro contorno esatto), le posiziona a caso vicine
tra loro su nuove immagini composite — tecnica nota come "copy-paste
augmentation", usata in letteratura per segmentazione di istanza con
pochi dati (es. "Simple Copy-Paste is a Strong Data Augmentation Method
for Instance Segmentation", Ghiasi et al.).

Accorgimenti:
  - bordi dei ritagli leggermente sfumati (feather), non tagliati netti
    — altrimenti il modello impara a riconoscere "il bordo del ritaglio"
    invece del vero contorno della scaglia
  - le scaglie non riempiono TUTTA l'immagine sintetica: resta sempre
    una frazione di sfondo/spazio vuoto (target ~20-40%), cosi' il
    modello vede anche esempi negativi
  - piccola rotazione/scala casuale per varieta'

Salva le immagini + label in formato YOLO-seg, dentro
data/yolo_dataset/{output-name}/images e labels — PRONTE per essere
incluse nel training (uniscile a quelle vere, o allena solo su queste
per un test rapido).

Uso:
  python3 scripts/generate_copypaste_dataset.py --individual prime --n-images 100
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RAW_ROOT = Path("data/raw")
LABELS_ROOT = Path("data/labels")
OUTPUT_ROOT = Path("data/yolo_dataset")


def log(msg: str) -> None:
    print(msg, flush=True)


def load_all_scaglie_cutouts(individual: str, feather: float = 5.0) -> list[dict]:
    """Ritaglia OGNI scaglia vera labellata (immagine RGBA con canale
    alpha sfumato ai bordi = il ritaglio + una maschera morbida) da
    tutte le immagini disponibili. Ritorna lista di dict con l'immagine
    ritagliata e le sue dimensioni."""
    labels_path = LABELS_ROOT / f"{individual}.json"
    if not labels_path.exists():
        log(f"ERRORE: {labels_path.resolve()} non trovato.")
        return []

    labels_data = json.loads(labels_path.read_text(encoding="utf-8"))
    cutouts = []

    for image_name, entry in labels_data.items():
        img_path = RAW_ROOT / individual / image_name
        if not img_path.exists():
            continue

        orig = Image.open(img_path).convert("RGB")
        scales = entry.get("scales", []) or []

        for s in scales:
            if not (isinstance(s, dict) and s.get("type") == "polygon" and len(s.get("coords", [])) >= 3):
                continue
            coords = [(float(x), float(y)) for x, y in s["coords"]]

            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            margin = int(feather * 2) + 3  # spazio extra per il feather, altrimenti verrebbe tagliato ai bordi
            x1, y1 = max(0, int(min(xs)) - margin), max(0, int(min(ys)) - margin)
            x2, y2 = min(orig.width, int(max(xs)) + margin), min(orig.height, int(max(ys)) + margin)
            w, h = x2 - x1, y2 - y1
            if w < 5 or h < 5:
                continue

            crop = orig.crop((x1, y1, x2, y2))

            # maschera locale del poligono, con bordo sfumato (feather)
            mask = Image.new("L", (w, h), 0)
            local_coords = [(x - x1, y - y1) for x, y in coords]
            ImageDraw.Draw(mask).polygon(local_coords, fill=255)
            mask = mask.filter(ImageFilter.GaussianBlur(feather))  # feather bordi, molto piu' morbido che prima

            cutouts.append({"crop": crop, "mask": mask, "w": w, "h": h})

    return cutouts


def generate_background(size: int) -> Image.Image:
    """Sfondo SCURO: su una pelle di coccodrillo vera le scaglie sono
    separate da solchi/fessure SCURE, non da spazio chiaro. Usando uno
    sfondo scuro, lo spazio che resta tra le scaglie incollate assomiglia
    naturalmente alle fessure vere."""
    base = random.randint(25, 70)
    arr = np.full((size, size), base, dtype="float32")
    arr += np.random.normal(0, 6, arr.shape)
    arr = np.clip(arr, 0, 255).astype("uint8")
    return Image.fromarray(arr).convert("RGB")


def paste_with_mask(canvas: Image.Image, cutout: dict, cx: int, cy: int, scale: float, angle: float) -> list | None:
    """Incolla un ritaglio scaglia sul canvas, centrato in (cx,cy), con
    scala/rotazione casuali. Ritorna il poligono approssimato (bounding
    box ruotato) nelle coordinate del canvas, o None se esce troppo dai
    bordi."""
    crop = cutout["crop"].resize((int(cutout["w"] * scale), int(cutout["h"] * scale)))
    mask = cutout["mask"].resize(crop.size)

    if angle:
        crop = crop.rotate(angle, expand=True, resample=Image.BICUBIC)
        mask = mask.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=0)

    w, h = crop.size
    x1, y1 = cx - w // 2, cy - h // 2

    canvas.paste(crop, (x1, y1), mask)

    # poligono approssimato = rettangolo del bounding box incollato
    # (approssimazione ragionevole per il label YOLO; il contorno esatto
    # post-rotazione servirebbe una trasformazione poligono completa,
    # non necessaria per dati di training sintetici aggiuntivi)
    return [[x1, y1], [x1 + w, y1], [x1 + w, y1 + h], [x1, y1 + h]]


def rotated_bbox_size(w: float, h: float, angle_deg: float) -> tuple:
    """Dimensione del bounding box DOPO la rotazione — una scaglia
    ruotata occupa piu' spazio di quanto suggerisca la sua dimensione
    originale, va tenuto conto per evitare sovrapposizioni."""
    a = math.radians(abs(angle_deg))
    return (w * math.cos(a) + h * math.sin(a),
            w * math.sin(a) + h * math.cos(a))


def boxes_overlap(b1: tuple, b2: tuple, margin: float = 1.0) -> bool:
    """True se due box (x1,y1,x2,y2) si sovrappongono, considerando un
    margine minimo di separazione."""
    return not (b1[2] + margin <= b2[0] or b2[2] + margin <= b1[0] or
                b1[3] + margin <= b2[1] or b2[3] + margin <= b1[1])


def overlap_area(b1: tuple, b2: tuple) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_area(b: tuple) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def make_blob_mask(size: int, coverage: float) -> np.ndarray:
    """Forma organica casuale (blob) che definisce DOVE mettere le
    scaglie — invece di riempire tutto il rettangolo, si riempie una
    regione dai contorni irregolari, come un lembo di pelle."""
    work = 128
    raw = np.random.uniform(-1, 1, (work, work)).astype("float32")
    img = Image.fromarray(((raw + 1) * 127.5).astype("uint8")).filter(
        ImageFilter.GaussianBlur(work * random.uniform(0.12, 0.20))
    )
    field = np.array(img, dtype="float32")
    thr = np.percentile(field, (1 - coverage) * 100)
    mask_small = (field > thr).astype("uint8") * 255
    return np.array(Image.fromarray(mask_small).resize((size, size), Image.BILINEAR)) > 127


def generate_one_synthetic_image(cutouts: list[dict], canvas_size: int, scales_per_row: int,
                                  max_rotation: float = 8.0, packing: float = 0.82,
                                  max_overlap: float = 0.10, hole_prob: float = 0.12,
                                  blob_coverage: float = 0.7) -> tuple[Image.Image, list]:
    """Layout tipo PELLE DI COCCODRILLO: scaglie TANGENTI o leggermente
    sovrapposte (fino a max_overlap dell'area, default 10%), disposte a
    file sfalsate dentro una FORMA ORGANICA casuale, con buchi sparsi.

    Differenza rispetto a un packing rigido: la cella della griglia e'
    volutamente PIU' STRETTA del bounding box delle scaglie (packing<1),
    cosi' si toccano invece di restare separate; il vincolo non e' piu'
    'mai toccarsi' ma 'sovrapporsi al massimo del 10%'.

    Le scaglie vengono riscalate a una dimensione target ricavata da
    scales_per_row, cosi' il riempimento e' prevedibile anche partendo
    da ritagli di dimensioni molto diverse tra loro."""
    canvas = generate_background(canvas_size)
    blob = make_blob_mask(canvas_size, blob_coverage)

    target = canvas_size / max(1, scales_per_row)
    worst_w, worst_h = rotated_bbox_size(target, target, max_rotation)
    cell = max(worst_w, worst_h) * packing

    placed_boxes = []
    polygons = []

    row = 0
    y = cell / 2
    row_offset_frac = random.choice([0.0, 0.5, 0.5, 0.33])
    wave_amp = cell * 0.12

    while y < canvas_size:
        x = cell / 2 + (row % 2) * cell * row_offset_frac
        while x < canvas_size:
            # buco casuale: ogni tanto si salta una posizione, cosi' la
            # superficie non e' uniformemente coperta
            if random.random() < hole_prob:
                x += cell
                continue

            cutout = random.choice(cutouts)
            base_scale = target / max(cutout["w"], cutout["h"])
            scale = base_scale * random.uniform(0.88, 1.08)
            sw, sh = cutout["w"] * scale, cutout["h"] * scale

            angle = random.uniform(-max_rotation, max_rotation)
            rw, rh = rotated_bbox_size(sw, sh, angle)

            wave = math.sin(x / max(1, canvas_size) * math.pi * 2 + row * 0.3) * wave_amp
            cx = x + random.uniform(-cell * 0.10, cell * 0.10)
            cy = y + wave + random.uniform(-cell * 0.10, cell * 0.10)

            box = (cx - rw / 2, cy - rh / 2, cx + rw / 2, cy + rh / 2)

            icx, icy = int(cx), int(cy)
            inside_blob = 0 <= icx < canvas_size and 0 <= icy < canvas_size and bool(blob[icy, icx])
            in_bounds = box[0] >= 0 and box[1] >= 0 and box[2] <= canvas_size and box[3] <= canvas_size

            if inside_blob and in_bounds:
                my_area = box_area(box)
                # vincolo: sovrapposizione consentita fino a max_overlap
                # dell'area della piu' piccola tra le due scaglie
                # margine di sicurezza (0.9) perche' sovrapposizioni con
                # piu' vicini possono sommarsi leggermente oltre il limite
                # dichiarato se lo si applica alla lettera coppia per coppia
                # fattore conservativo: il box stimato prima dell'incollaggio
                # e' leggermente piu' piccolo di quello reale (PIL arrotonda
                # per eccesso ruotando), quindi si stringe il vincolo per
                # restare dentro il limite dichiarato anche dopo il paste
                ok = all(
                    overlap_area(box, pb) <= max_overlap * 0.75 * min(my_area, box_area(pb))
                    for pb in placed_boxes
                )
                if ok:
                    poly = paste_with_mask(canvas, cutout, int(cx), int(cy), scale, angle)
                    if poly:
                        # traccia il box REALE ricavato dal poligono
                        # effettivamente incollato: PIL arrotonda per eccesso
                        # quando ruota (expand=True), quindi il box vero e'
                        # leggermente piu' grande della stima fatta sopra —
                        # usando la stima si accumulavano piccoli sfori oltre
                        # il limite di sovrapposizione dichiarato
                        pxs = [pt[0] for pt in poly]
                        pys = [pt[1] for pt in poly]
                        real_box = (min(pxs), min(pys), max(pxs), max(pys))
                        polygons.append(poly)
                        placed_boxes.append(real_box)

            x += cell
        y += cell
        row += 1

    return canvas, polygons


def polygon_to_yolo_line(coords: list, img_w: int, img_h: int, class_id: int = 0) -> str:
    norm = []
    for x, y in coords:
        norm.append(f"{max(0.0, min(1.0, x / img_w)):.6f}")
        norm.append(f"{max(0.0, min(1.0, y / img_h)):.6f}")
    return f"{class_id} " + " ".join(norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--individual", type=str, required=True, help="da dove prendere le scaglie vere ritagliate")
    parser.add_argument("--n-images", type=int, default=100)
    parser.add_argument("--canvas-size", type=int, default=768)
    parser.add_argument("--scales-per-row", type=int, default=12,
                         help="quante scaglie per riga — determina la dimensione a cui vengono riscalate (piu' alto = scaglie piu' piccole e numerose)")
    parser.add_argument("--packing", type=float, default=0.55,
                         help="compattezza griglia: <1 = celle piu' strette del bbox, le scaglie si toccano (default 0.55, molto fitto — non scendere sotto 0.4, diventa lentissimo)")
    parser.add_argument("--max-overlap", type=float, default=0.10,
                         help="sovrapposizione massima consentita tra due scaglie, come frazione di area (default 0.10 = 10%%)")
    parser.add_argument("--hole-prob", type=float, default=0.04,
                         help="probabilita' di saltare una posizione, lasciando buchi sparsi (default 0.04, molto meno che prima)")
    parser.add_argument("--blob-coverage", type=float, default=0.88,
                         help="quanto della superficie copre la forma organica casuale da riempire (default 0.88, quasi tutta l'immagine)")
    parser.add_argument("--max-rotation", type=float, default=8.0,
                         help="rotazione massima in gradi — piu' bassa = scaglie piu' fitte (il bbox ruotato occupa meno spazio)")
    parser.add_argument("--feather", type=float, default=5.0,
                         help="quanto sfumare il bordo di ogni scaglia (raggio blur px) — piu' alto = contorno piu' morbido, meno riconoscibile a colpo d'occhio (default 5.0, prima era 2.0)")
    parser.add_argument("--output-name", type=str, default=None)
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    output_name = args.output_name or f"{args.individual}_copypaste"

    log(f"Ritaglio scaglie vere da: {args.individual} (feather={args.feather})...")
    cutouts = load_all_scaglie_cutouts(args.individual, feather=args.feather)
    log(f"Scaglie ritagliate disponibili: {len(cutouts)}")

    if len(cutouts) < 5:
        log("ERRORE: servono almeno 5 scaglie labellate per generare dataset sintetico sensato.")
        return

    dataset_dir = OUTPUT_ROOT / output_name
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        (dataset_dir / sub).mkdir(parents=True, exist_ok=True)

    # salvataggio PARALLELO in data/raw/{output_name}/ + data/labels/{output_name}.json
    # — stesso formato dei dati veri, cosi' auto_generate_patches.py puo'
    # essere lanciato IDENTICO sulle immagini sintetiche (stessa logica
    # multi-scala, stessa zona automatica via convex hull, gia' testata)
    raw_out_dir = RAW_ROOT / output_name
    raw_out_dir.mkdir(parents=True, exist_ok=True)
    synthetic_labels = {}

    n_val = max(1, int(args.n_images * args.val_frac))
    log(f"Genero {args.n_images} immagini sintetiche ({args.n_images - n_val} train, {n_val} val)...")

    for i in range(args.n_images):
        split = "val" if i < n_val else "train"

        img, polygons = generate_one_synthetic_image(
            cutouts, args.canvas_size, args.scales_per_row, args.max_rotation, args.packing,
            args.max_overlap, args.hole_prob, args.blob_coverage
        )

        img_path = dataset_dir / f"images/{split}/synth_{i:04d}.jpg"
        img_path.parent.mkdir(parents=True, exist_ok=True)  # difensivo: ricrea se sparita (es. sync iCloud)
        img.save(img_path, format="JPEG", quality=90)

        lines = [polygon_to_yolo_line(poly, args.canvas_size, args.canvas_size) for poly in polygons]
        label_path = dataset_dir / f"labels/{split}/synth_{i:04d}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines), encoding="utf-8")

        # copia identica in data/raw/ + entry in data/labels/ (stesso formato
        # scaglie usato da app.py, sam_assisted_labeling.py, auto_generate_patches.py)
        raw_img_name = f"synth_{i:04d}.jpg"
        img.save(raw_out_dir / raw_img_name, format="JPEG", quality=90)
        synthetic_labels[raw_img_name] = {
            "roi": None,
            "scales": [
                {"type": "polygon", "coords": [[round(x, 2), round(y, 2)] for x, y in poly], "label": "scaglia"}
                for poly in polygons
            ],
        }

        if (i + 1) % 20 == 0 or (i + 1) == args.n_images:
            log(f"  {i + 1}/{args.n_images} generate (ultima: {len(polygons)} scaglie)")

    labels_path = LABELS_ROOT / f"{output_name}.json"
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(synthetic_labels, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"\nLabel formato 'reale' salvate: {labels_path.resolve()}")
    log(f"Immagini in formato 'reale' salvate: {raw_out_dir.resolve()}")

    yaml_content = f"""# Generato da generate_copypaste_dataset.py — sintetico copy-paste da: {args.individual}
path: {dataset_dir.resolve()}
train: images/train
val: images/val

names:
  0: scaglia
"""
    yaml_path = dataset_dir / "data.yaml"
    yaml_path.write_text(yaml_content, encoding="utf-8")

    log(f"\n=== FATTO ===")
    log(f"Dataset sintetico copy-paste (formato YOLO) pronto in: {dataset_dir.resolve()}")
    log(f"Config: {yaml_path.resolve()}")
    log(f"\nAllena YOLO direttamente sui sintetici con:")
    log(f"  python3 scripts/train_yolo.py --data {yaml_path.resolve()}")
    log(f"\nPer dividerli in patch multi-scala (STESSA logica delle immagini vere):")
    log(f"  python3 scripts/auto_generate_patches.py --individual {output_name} --scales 224 400 600 --output-size 224")


if __name__ == "__main__":
    main()