"""
Applica YOLO11-seg a immagini INTERE usando DIVIDE ET IMPERA MULTI-SCALA:
- piu' dimensioni di tile sulla STESSA immagine (es. 512 e 768px)
- un passaggio EXTRA di tile centrati esattamente sulle giunzioni tra i
  tile della griglia base (non ci si affida solo all'overlap: un tile
  dedicato viene posizionato proprio a cavallo di ogni confine, incroci
  compresi)

Pulizia dei duplicati in DUE fasi:
  1. Se due box si sovrappongono per piu' del 40% dell'area del piu'
     piccolo, il piccolo viene ASSORBITO (scartato) in quello grande.
  2. Per le coppie che restano ma i cui SEGMENTI (maschere) ancora si
     toccano, non si scarta nulla: si RITAGLIA via dal poligono piu'
     piccolo solo la porzione in comune con quello piu' grande (che
     resta intatto) — richiede 'shapely' per il ritaglio geometrico.

Output: immagine ORIGINALE intera con maschere/box disegnati sopra per
ogni scaglia trovata, salvata in JPEG.

Uso:
  pip install shapely
  python3 scripts/predict_yolo_tiled.py --model models/yolo_runs/prime_TIMESTAMP/weights/best.pt --folder data/test_images
  python3 scripts/predict_yolo_tiled.py --model ... --yolo-dataset data/yolo_dataset/prime --tile-sizes 512 768 --conf 0.15
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def log(msg: str) -> None:
    print(msg, flush=True)


def compute_tile_grid(img_w: int, img_h: int, tile_size: int, overlap: int) -> list:
    """Griglia base di tile sovrapposti — copre l'intera immagine."""
    stride = tile_size - overlap
    xs = list(range(0, max(1, img_w - tile_size + 1), stride)) or [0]
    ys = list(range(0, max(1, img_h - tile_size + 1), stride)) or [0]
    if xs[-1] + tile_size < img_w:
        xs.append(img_w - tile_size)
    if ys[-1] + tile_size < img_h:
        ys.append(img_h - tile_size)
    return [(x, y, min(x + tile_size, img_w), min(y + tile_size, img_h)) for y in ys for x in xs]


def compute_junction_tiles(img_w: int, img_h: int, base_tile_size: int, overlap: int,
                           junction_tile_size: int, max_junction_tiles: int = 300) -> list:
    """Tile EXTRA centrati esattamente sui punti di giunzione (confine)
    tra i tile della griglia base — verticali, orizzontali, e ad incrocio
    (dove 4 tile si toccano). Non sostituisce l'overlap, lo rinforza.

    TETTO DI SICUREZZA: con tile base piccoli e overlap grande, lo stride
    puo' diventare minuscolo, generando MIGLIAIA di punti di giunzione
    (visto in pratica: 1993 tile da 896px, 261mila rilevamenti solo da
    quelli). Se il conteggio supera max_junction_tiles, si campiona a
    caso fino al tetto — meglio una copertura giunzioni piu' rada che
    un'esplosione combinatoria che rende tutto lentissimo."""
    stride = base_tile_size - overlap
    xs = list(range(0, max(1, img_w - base_tile_size + 1), stride)) or [0]
    ys = list(range(0, max(1, img_h - base_tile_size + 1), stride)) or [0]
    if xs[-1] + base_tile_size < img_w:
        xs.append(img_w - base_tile_size)
    if ys[-1] + base_tile_size < img_h:
        ys.append(img_h - base_tile_size)

    seam_xs = [(xs[i] + base_tile_size + xs[i + 1]) / 2 for i in range(len(xs) - 1)]
    seam_ys = [(ys[i] + base_tile_size + ys[i + 1]) / 2 for i in range(len(ys) - 1)]

    junction_centers = []
    for sx in seam_xs:
        for y in ys:
            junction_centers.append((sx, y + base_tile_size / 2))
    for x in xs:
        for sy in seam_ys:
            junction_centers.append((x + base_tile_size / 2, sy))
    for sx in seam_xs:
        for sy in seam_ys:
            junction_centers.append((sx, sy))

    if len(junction_centers) > max_junction_tiles:
        log(f"  ATTENZIONE: {len(junction_centers)} punti di giunzione calcolati, "
            f"oltre il tetto di sicurezza ({max_junction_tiles}) — campiono a caso per restare veloce.")
        random.shuffle(junction_centers)
        junction_centers = junction_centers[:max_junction_tiles]

    tiles = []
    half = junction_tile_size / 2
    for cx, cy in junction_centers:
        x1 = max(0, min(img_w - junction_tile_size, cx - half))
        y1 = max(0, min(img_h - junction_tile_size, cy - half))
        x2, y2 = x1 + junction_tile_size, y1 + junction_tile_size
        if x2 <= img_w and y2 <= img_h:
            tiles.append((int(x1), int(y1), int(x2), int(y2)))
    return tiles


def box_overlap_area(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def box_area(a: tuple) -> float:
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def overlap_fraction_of_smaller(a: tuple, b: tuple) -> float:
    """Frazione di sovrapposizione calcolata sull'area del box PIU'
    PICCOLO tra i due — se il piccolo e' quasi tutto dentro il grande,
    questo valore e' alto anche se l'IoU (che divide per l'unione) non
    lo sarebbe."""
    inter = box_overlap_area(a, b)
    smaller_area = min(box_area(a), box_area(b))
    return inter / smaller_area if smaller_area > 0 else 0.0


def merge_by_absorption(detections: list, overlap_threshold: float = 0.4) -> list:
    """Se due box si sovrappongono per piu' di 'overlap_threshold' (default
    40%) dell'AREA DEL PIU' PICCOLO, il piccolo viene ASSORBITO in quello
    piu' grande (scartato) — non si sceglie in base alla confidenza, si
    sceglie in base alla DIMENSIONE: il box grande vince sempre.
    Si processa dal piu' grande al piu' piccolo, cosi' un box gia' tenuto
    e' sempre >= in area di ogni candidato successivo."""
    detections_sorted = sorted(detections, key=lambda d: box_area(d["box"]), reverse=True)
    kept = []
    for det in detections_sorted:
        absorbed = any(
            overlap_fraction_of_smaller(det["box"], k["box"]) > overlap_threshold
            for k in kept
        )
        if not absorbed:
            kept.append(det)
    return kept


def run_tiles_on_image(model, img_arr: np.ndarray, tiles: list, conf: float, batch_size: int, tag: str) -> list:
    """Esegue il modello su una lista di tile (a blocchi, per evitare OOM),
    ritorna rilevamenti in coordinate IMMAGINE INTERA — sia il box (usato
    per il calcolo NMS) sia il POLIGONO MASCHERA preciso (usato per il
    disegno finale, molto piu' fedele di un rettangolo)."""
    detections = []
    for start in range(0, len(tiles), batch_size):
        chunk_tiles = tiles[start:start + batch_size]
        chunk_arrays = [img_arr[ty1:ty2, tx1:tx2] for (tx1, ty1, tx2, ty2) in chunk_tiles]

        results_chunk = model(chunk_arrays, save=False, conf=conf, verbose=False)

        for (tx1, ty1, tx2, ty2), r in zip(chunk_tiles, results_chunk):
            if r.boxes is None:
                continue

            # r.masks.xy: un array di punti (x,y) per ogni istanza, gia'
            # in coordinate del tile — None se il modello non e' -seg
            mask_polys = r.masks.xy if r.masks is not None else None

            for i, box in enumerate(r.boxes):
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]

                polygon = None
                if mask_polys is not None and i < len(mask_polys):
                    polygon = [[float(px) + tx1, float(py) + ty1] for px, py in mask_polys[i]]

                detections.append({
                    "box": (x1 + tx1, y1 + ty1, x2 + tx1, y2 + ty1),
                    "polygon": polygon,
                    "conf": float(box.conf[0]),
                    "source": tag,
                })
    return detections


def chaikin_smooth(points: list, iterations: int = 2) -> list:
    """Ammorbidisce un poligono (Chaikin's corner-cutting) — sostituisce
    ogni spigolo con due punti piu' vicini ai lati, arrotondando le
    forme senza bisogno di librerie esterne. Con 2 iterazioni gia'
    ammorbidisce visibilmente contorni a zig-zag/blocchi mantenendo la
    forma generale. Serve SOLO per il disegno finale — non tocca la
    geometria usata per i calcoli di overlap/ritaglio."""
    if len(points) < 3:
        return points
    pts = points
    for _ in range(iterations):
        new_pts = []
        n = len(pts)
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            q = [0.75 * p0[0] + 0.25 * p1[0], 0.75 * p0[1] + 0.25 * p1[1]]
            r = [0.25 * p0[0] + 0.75 * p1[0], 0.25 * p0[1] + 0.75 * p1[1]]
            new_pts.extend([q, r])
        pts = new_pts
    return pts


def box_to_rect_points(box: tuple) -> list:
    x1, y1, x2, y2 = box
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def clip_overlapping_polygons(detections: list) -> list:
    """UNICA fase di pulizia per le sovrapposizioni: NESSUN rilevamento
    viene scartato per intero in base a una soglia percentuale. Si
    RITAGLIA via dal poligono/box piu' piccolo solo la porzione che sta
    SOPRA quello piu' grande (che resta intatto) — si eliminano solo i
    vertici della parte in comune, il resto della forma resta. Una
    forma sparisce SOLO se e' completamente coperta al 100% da una piu'
    grande (nessuna parte propria residua), mai per una soglia arbitraria.

    Funziona sia sulle maschere (poligono preciso) sia sui rilevamenti
    senza maschera (fallback box-only): in quel caso si ritaglia il
    rettangolo del box come se fosse il poligono.

    OTTIMIZZAZIONE: usa un indice spaziale (STRtree) per confrontare ogni
    rilevamento SOLO con quelli vicini geometricamente, non con tutti
    quelli gia' tenuti — con migliaia di rilevamenti, il confronto
    "con tutti" (O(n^2)) diventa lentissimo (visto in pratica: ~15000
    rilevamenti, centinaia di milioni di confronti nel caso peggiore).
    Con l'indice spaziale, ogni rilevamento controlla solo una manciata
    di vicini, non l'intera lista."""
    try:
        from shapely.geometry import Polygon
        from shapely.validation import make_valid
        from shapely.strtree import STRtree
    except ImportError:
        log("ATTENZIONE: 'shapely' non installato, salto il ritaglio poligoni sovrapposti.")
        log("  Installa con: pip install shapely")
        return detections

    # dal piu' grande al piu' piccolo: i grandi restano intatti, i
    # piccoli vengono ritagliati contro quelli gia' tenuti
    order = sorted(range(len(detections)), key=lambda i: box_area(detections[i]["box"]), reverse=True)

    kept_shapes: list = []  # poligoni shapely GIA' tenuti (originali, mai ritagliati)
    tree = None
    pending_recent: list = []  # forme aggiunte dall'ultima ricostruzione indice — controllate sempre, per intero (lista piccola e limitata, economico)
    REBUILD_EVERY = 50  # ricostruisce l'indice spaziale ogni N inserimenti, non ad ognuno (compromesso costo/precisione)

    result = []

    for idx in order:
        det = detections[idx]
        source_points = det.get("polygon") if det.get("polygon") and len(det["polygon"]) >= 3 else box_to_rect_points(det["box"])

        try:
            poly = Polygon(source_points)
            if not poly.is_valid:
                poly = make_valid(poly)
        except Exception:
            result.append(det)
            continue

        if tree is None and kept_shapes:
            tree = STRtree(kept_shapes)

        candidates = []
        if tree is not None:
            candidate_idxs = tree.query(poly)
            candidates = [kept_shapes[i] for i in candidate_idxs]
        # SEMPRE controllate anche le forme aggiunte dopo l'ultima
        # ricostruzione dell'indice (lista piccola, max REBUILD_EVERY
        # elementi — economico, e garantisce che nessun overlap recente
        # venga perso solo perche' l'indice non e' ancora stato aggiornato)
        candidates.extend(pending_recent)

        clipped = poly
        for kp in candidates:
            if clipped.is_empty:
                break
            if clipped.intersects(kp):
                clipped = clipped.difference(kp)

        if clipped.is_empty:
            continue  # completamente coperto (100%) da forme piu' grandi, nessuna parte propria residua

        # se il ritaglio ha spezzato il poligono in piu' pezzi, tieni il
        # pezzo piu' grande (rappresenta ancora la stessa scaglia)
        if clipped.geom_type == "MultiPolygon":
            clipped = max(clipped.geoms, key=lambda g: g.area)
        if clipped.geom_type != "Polygon" or clipped.is_empty:
            continue

        new_det = dict(det)
        new_det["polygon"] = [[float(x), float(y)] for x, y in clipped.exterior.coords]
        result.append(new_det)

        # nella lista dei "gia' tenuti" resta il poligono ORIGINALE (non
        # ritagliato) — solo i piu' piccoli successivi vengono tagliati
        # contro questo, mai il contrario
        kept_shapes.append(poly)
        pending_recent.append(poly)
        if len(pending_recent) >= REBUILD_EVERY:
            tree = STRtree(kept_shapes)  # ricostruisce includendo tutte le forme, pending compreso
            pending_recent = []
    return result


def filter_low_contrast_detections(detections: list, img_arr: np.ndarray, min_std: float) -> tuple:
    """Scarta i rilevamenti la cui zona sottostante e' troppo UNIFORME
    (poco contrasto) — sintomo di falso positivo su sfondo/bordo pelle
    senza vera texture di scaglia, non di un duplicato da fondere.
    Il NMS non puo' risolvere questo: sono rilevamenti in posizioni
    diverse, genuinamente separati, ma tutti sbagliati sulla stessa
    zona vuota."""
    gray = np.array(Image.fromarray(img_arr).convert("L"), dtype="float32")
    img_h, img_w = gray.shape

    kept = []
    discarded = 0
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        xi1, yi1 = max(0, int(x1)), max(0, int(y1))
        xi2, yi2 = min(img_w, int(x2)), min(img_h, int(y2))
        if xi2 <= xi1 or yi2 <= yi1:
            discarded += 1
            continue
        patch_std = gray[yi1:yi2, xi1:xi2].std()
        if patch_std >= min_std:
            kept.append(det)
        else:
            discarded += 1
    return kept, discarded


def process_image(model, img_path: Path, tile_sizes: list, overlap: int, conf: float,
                   nms_iou: float, batch_size: int, output_dir: Path, jpeg_quality: int,
                   junction_tile_size: int, min_local_contrast: float = 8.0, smoothing_iterations: int = 2,
                   max_junction_tiles: int = 300) -> dict:
    orig = Image.open(img_path).convert("RGB")
    img_arr = np.array(orig)
    img_w, img_h = orig.size

    log(f"\n--- {img_path.name} — {img_w}x{img_h}px ---")

    all_detections = []
    stats_per_source = {}

    for tile_size in tile_sizes:
        tiles = compute_tile_grid(img_w, img_h, tile_size, overlap)
        tag = f"scale{tile_size}"
        log(f"  [{tag}] {len(tiles)} tile...")
        dets = run_tiles_on_image(model, img_arr, tiles, conf, batch_size, tag)
        all_detections.extend(dets)
        stats_per_source[tag] = len(dets)
        log(f"  [{tag}] {len(dets)} rilevamenti grezzi")

    # passaggio extra: tile centrati sulle giunzioni della griglia PIU' FINE
    # (quella con overlap piu' stretto in proporzione, di solito la piu' piccola)
    base_tile_size = min(tile_sizes)
    junction_tiles = compute_junction_tiles(img_w, img_h, base_tile_size, overlap, junction_tile_size, max_junction_tiles)
    log(f"  [giunzioni] {len(junction_tiles)} tile centrati sui confini (dimensione {junction_tile_size}px)...")
    junction_dets = run_tiles_on_image(model, img_arr, junction_tiles, conf, batch_size, "giunzione")
    all_detections.extend(junction_dets)
    stats_per_source["giunzione"] = len(junction_dets)
    log(f"  [giunzioni] {len(junction_dets)} rilevamenti grezzi")

    log(f"  Totale rilevamenti grezzi (tutte le scale + giunzioni): {len(all_detections)}")

    n_with_mask = sum(1 for d in all_detections if d.get("polygon"))
    n_fallback_box = len(all_detections) - n_with_mask
    log(f"  Di cui con maschera reale: {n_with_mask} ({100*n_with_mask/max(1,len(all_detections)):.0f}%) "
        f"— fallback rettangolo (nessuna maschera dal modello): {n_fallback_box} "
        f"({100*n_fallback_box/max(1,len(all_detections)):.0f}%)")
    if n_fallback_box > len(all_detections) * 0.3:
        log(f"  ATTENZIONE: piu' del 30% dei rilevamenti non ha una maschera reale — "
            f"e' probabile che l'aspetto \"geometrico/a blocchi\" nell'immagine finale venga da qui.")

    n_before_clip = len(all_detections)
    final_detections = clip_overlapping_polygons(all_detections)
    log(f"  Rilevamenti dopo ritaglio sovrapposizioni: {len(final_detections)} "
        f"(nessuno scartato per soglia percentuale — solo {n_before_clip - len(final_detections)} "
        f"completamente coperti al 100% da altri, tutti gli altri mantengono la loro parte propria)")

    final_detections, n_discarded_contrast = filter_low_contrast_detections(final_detections, img_arr, min_local_contrast)
    log(f"  Rilevamenti dopo filtro contrasto locale (min_std={min_local_contrast}): {len(final_detections)} "
        f"({n_discarded_contrast} scartati come falsi positivi su zone uniformi/sfondo)")

    overlay = orig.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    for det in final_detections:
        if det.get("polygon") and len(det["polygon"]) >= 3:
            # smoothing SOLO per il disegno — la geometria usata sopra per
            # ritaglio/overlap resta quella precisa, non tocchiamo quella
            smoothed = chaikin_smooth(det["polygon"], iterations=smoothing_iterations) if smoothing_iterations > 0 else det["polygon"]
            pts = [(x, y) for x, y in smoothed]
            draw.polygon(pts, outline=(255, 140, 0, 255), width=2, fill=(255, 140, 0, 60))
        else:
            # fallback: nessuna maschera disponibile per questo rilevamento
            x1, y1, x2, y2 = det["box"]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 140, 0, 255), width=2)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{img_path.stem}_yolo_tiled.jpg"
    overlay.convert("RGB").save(out_path, format="JPEG", quality=jpeg_quality, optimize=True)
    log(f"  Salvato: {out_path.resolve()}")

    return {
        "image": img_path.name,
        "tile_sizes": tile_sizes,
        "n_junction_tiles": len(junction_tiles),
        "raw_detections_per_source": stats_per_source,
        "n_raw_total": len(all_detections),
        "n_final_detections": len(final_detections),
        "output": str(out_path.resolve()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="path a weights/best.pt")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--yolo-dataset", type=str, default=None,
                         help="cartella data/yolo_dataset/{run}/ — prende le immagini da images/train + images/val")
    parser.add_argument("--tile-sizes", type=int, nargs="+", default=[512, 768],
                         help="piu' dimensioni tile sulla stessa immagine, es: --tile-sizes 512 768 1024")
    parser.add_argument("--junction-tile-size", type=int, default=None,
                         help="dimensione dei tile centrati sulle giunzioni (default: la PIU' PICCOLA tra --tile-sizes — economica)")
    parser.add_argument("--max-junction-tiles", type=int, default=300,
                         help="tetto massimo di tile giunzione — con overlap grande su tile piccoli i punti di giunzione "
                              "possono esplodere a migliaia (visto in pratica: 1993 tile, 261mila rilevamenti). "
                              "Oltre questo numero si campiona a caso (default 300).")
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--nms-iou", type=float, default=0.4,
                         help="NON PIU' USATO: le sovrapposizioni ora vengono sempre ritagliate geometricamente "
                              "(clip_overlapping_polygons), mai scartate per soglia percentuale. Parametro tenuto "
                              "solo per compatibilita' con comandi gia' scritti, non ha piu' effetto.")
    parser.add_argument("--min-local-contrast", type=float, default=8.0,
                         help="scarta rilevamenti su zone troppo uniformi (falsi positivi su sfondo/bordo pelle) — deviazione standard minima dei pixel nel box, 0-255. Alza se vedi ancora falsi positivi su sfondo, abbassa se scarta scaglie vere poco contrastate")
    parser.add_argument("--smoothing", type=int, default=2,
                         help="quante iterazioni di ammorbidimento contorno (Chaikin) applicare al disegno finale — "
                              "0 disattiva, valori piu' alti (3-4) ammorbidiscono di piu' ma arrotondano anche dettagli veri. "
                              "Non tocca la geometria usata per i calcoli, solo l'aspetto visivo.")
    parser.add_argument("--batch", type=int, default=2, help="tile processati insieme — abbassa se OOM")
    parser.add_argument("--output-dir", type=str, default="data/yolo_predictions_full")
    parser.add_argument("--jpeg-quality", type=int, default=75)
    args = parser.parse_args()

    n_sources = sum(1 for x in (args.image, args.folder, args.yolo_dataset) if x)
    if n_sources == 0:
        log("ERRORE: specifica --image, --folder oppure --yolo-dataset")
        return
    if n_sources > 1:
        log("ERRORE: usa solo uno tra --image, --folder, --yolo-dataset")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        log("ERRORE: manca il pacchetto. Installa con: pip install ultralytics")
        return

    model_path = Path(args.model)
    if not model_path.exists():
        log(f"ERRORE: modello non trovato: {model_path.resolve()}")
        return

    log(f"Carico modello: {model_path.resolve()}")
    model = YOLO(str(model_path))

    if args.image:
        images = [Path(args.image)]
    elif args.yolo_dataset:
        dataset_root = Path(args.yolo_dataset)
        if not dataset_root.exists():
            log(f"ERRORE: cartella non trovata: {dataset_root.resolve()}")
            return
        images = []
        for split in ["train", "val"]:
            split_dir = dataset_root / "images" / split
            if split_dir.exists():
                found = sorted(p for p in split_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
                log(f"  images/{split}: {len(found)} immagini")
                images.extend(found)
            else:
                log(f"  images/{split}: cartella non trovata, salto.")
    else:
        folder = Path(args.folder)
        if not folder.exists():
            log(f"ERRORE: cartella non trovata: {folder.resolve()}")
            return
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)

    if not images:
        log("Nessuna immagine trovata.")
        return

    tile_sizes = sorted(args.tile_sizes)
    junction_tile_size = args.junction_tile_size or min(tile_sizes)

    log(f"Immagini da processare: {len(images)}")
    log(f"Scale tile: {tile_sizes} — overlap: {args.overlap}px — tile giunzione: {junction_tile_size}px")
    log(f"conf: {args.conf} — NMS IoU: {args.nms_iou}")

    output_dir = Path(args.output_dir)
    all_results = []
    for img_path in images:
        result = process_image(
            model, img_path, tile_sizes, args.overlap, args.conf,
            args.nms_iou, args.batch, output_dir, args.jpeg_quality, junction_tile_size,
            args.min_local_contrast, args.smoothing, args.max_junction_tiles,
        )
        all_results.append(result)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    total_scales = sum(r["n_final_detections"] for r in all_results)
    log(f"\n=== FATTO ===")
    log(f"Immagini processate: {len(all_results)}")
    log(f"Scaglie totali trovate (dopo fusione): {total_scales}")
    log(f"Immagini con box disegnati: {output_dir.resolve()}/")
    log(f"Riepilogo: {summary_path.resolve()}")


if __name__ == "__main__":
    main()