"""
GUI di labeling assistita da SAM (Segment Anything) — TUTTO IN UN FILE.

Quattro modalita' (radio in sidebar):
  1. SAM (click singolo) — click su un punto, SAM segmenta quella scaglia
  2. SAM (zona rettangolare) — disegna un rettangolo, SAM automatico li' dentro
  3. Manuale (poligono) — click per ogni vertice, poi "Chiudi poligono"
  4. Elimina (click su figura) — click dentro una forma per cancellarla

"Suggerisci tutte" applica SAM con logica DIVIDE ET IMPERA a TILE
SOVRAPPOSTI: l'immagine viene divisa in tile (default 512px, overlap
150px), SAM gira su ognuno separatamente, l'overlap cattura le forme
tagliate dai bordi, i duplicati tra tile vengono scartati.

Salva in data/labels/{individuo}.json — stesso formato di app.py.
Salvataggio automatico ad ogni modifica.

Uso:
  pip install segment-anything --break-system-packages
  curl -L -o models/sam_vit_b.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
  streamlit run scripts/sam_assisted_labeling.py
"""

import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas

RAW_ROOT = Path("data/raw")
LABELS_ROOT = Path("data/labels")
LABELS_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]
POINT_RADIUS = 5
VIEWPORT_W = 1000
VIEWPORT_H = 680
SAM_CHECKPOINT = "models/sam_vit_b.pth"
SAM_MODEL_TYPE = "vit_b"
SAM_DEVICE = "cpu"  # MPS non supportato da SAM (float64)

st.set_page_config(page_title="Labeling assistito SAM", layout="wide")


# ================================================================ SAM

@st.cache_resource(show_spinner="Carico SAM (una sola volta)...")
def load_sam():
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

    ckpt = Path(SAM_CHECKPOINT)
    if not ckpt.exists():
        return None, None, None

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(ckpt))
    sam.to(SAM_DEVICE)
    predictor = SamPredictor(sam)
    auto_gen = SamAutomaticMaskGenerator(
        sam, points_per_side=24, pred_iou_thresh=0.86,
        stability_score_thresh=0.92, min_mask_region_area=50,
    )
    return sam, predictor, auto_gen


# ================================================================ geometria

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


def polygon_area(poly: list) -> float:
    n = len(poly)
    area = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def compute_tile_grid(img_w: int, img_h: int, tile_size: int, overlap: int) -> list:
    # Griglia con ultimo tile "ancorato" a destra/basso: evita di lasciare
    # strisce non coperte quando dimensioni immagine non sono multipli esatti.
    stride = tile_size - overlap
    xs = list(range(0, max(1, img_w - tile_size + 1), stride)) or [0]
    ys = list(range(0, max(1, img_h - tile_size + 1), stride)) or [0]
    if xs[-1] + tile_size < img_w:
        xs.append(img_w - tile_size)
    if ys[-1] + tile_size < img_h:
        ys.append(img_h - tile_size)
    return [(x, y, min(x + tile_size, img_w), min(y + tile_size, img_h)) for y in ys for x in xs]


def rasterize_shapes(shapes: list, width: int, height: int) -> np.ndarray:
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for poly in shapes:
        draw.polygon([(x, y) for x, y in poly], fill=255)
    return np.array(canvas) > 0


def polygon_bbox(poly: list, img_w: int, img_h: int, pad: int = 2) -> tuple:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (
        max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
        min(img_w, int(max(xs)) + pad + 1), min(img_h, int(max(ys)) + pad + 1),
    )


def rasterize_polygon_local(poly: list, bbox: tuple) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return np.zeros((0, 0), dtype=bool)
    canvas = Image.new("L", (w, h), 0)
    ImageDraw.Draw(canvas).polygon([(x - x1, y - y1) for x, y in poly], fill=255)
    return np.array(canvas) > 0


def overlap_fraction_fast(poly_full: list, existing_mask: np.ndarray, img_w: int, img_h: int) -> tuple:
    """Overlap calcolato solo sul bounding box locale (177x piu' veloce
    che rasterizzare l'intera immagine per ogni candidato)."""
    bbox = polygon_bbox(poly_full, img_w, img_h)
    local_mask = rasterize_polygon_local(poly_full, bbox)
    if local_mask.size == 0:
        return 0.0, bbox, local_mask
    x1, y1, x2, y2 = bbox
    cand_area = local_mask.sum()
    if cand_area == 0:
        return 0.0, bbox, local_mask
    overlap = (local_mask & existing_mask[y1:y2, x1:x2]).sum() / cand_area
    return float(overlap), bbox, local_mask


def parse_last_rect_from_canvas(canvas_json: dict, scale: float) -> list | None:
    if not canvas_json:
        return None
    rects = [o for o in canvas_json.get("objects", []) if o.get("type") == "rect"]
    if not rects:
        return None
    obj = rects[-1]
    left, top = float(obj.get("left", 0)), float(obj.get("top", 0))
    w = float(obj.get("width", 0)) * float(obj.get("scaleX", 1))
    h = float(obj.get("height", 0)) * float(obj.get("scaleY", 1))
    x1, y1 = left / scale, top / scale
    x2, y2 = (left + w) / scale, (top + h) / scale
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


# ================================================================ dati

def list_individuals() -> list[str]:
    if not RAW_ROOT.exists():
        return []
    return sorted([p.name for p in RAW_ROOT.iterdir() if p.is_dir()])


@st.cache_data(show_spinner=False)
def list_images(individual: str) -> list[str]:
    folder = RAW_ROOT / individual
    found = []
    for ext in IMAGE_EXTENSIONS:
        found.extend(folder.glob(ext))
    return sorted({p.name for p in found})


@st.cache_data(show_spinner=False)
def load_image_cached(path_str: str, mtime: float) -> Image.Image:
    return Image.open(path_str).convert("RGB")


@st.cache_data(show_spinner=False)
def load_image_array(path_str: str, mtime: float) -> np.ndarray:
    return np.array(Image.open(path_str).convert("RGB"))


@st.cache_data(show_spinner=False)
def resized_bg_cached(path_str: str, mtime: float, w: int, h: int) -> Image.Image:
    return Image.open(path_str).convert("RGB").resize((w, h))


def load_individual_labels(individual: str) -> dict:
    path = LABELS_ROOT / f"{individual}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def persist_shapes(individual: str, image_name: str, shapes: list) -> None:
    labels_data = load_individual_labels(individual)
    entry = labels_data.get(image_name, {})
    entry["scales"] = [
        {"type": "polygon", "coords": [[round(x, 2), round(y, 2)] for x, y in poly], "label": "scaglia"}
        for poly in shapes
    ]
    labels_data[image_name] = entry
    (LABELS_ROOT / f"{individual}.json").write_text(
        json.dumps(labels_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def state_key(individual: str, image: str) -> str:
    return f"shapes_{individual}_{image}"


def get_shapes(individual: str, image: str) -> list:
    key = state_key(individual, image)
    if key not in st.session_state:
        entry = load_individual_labels(individual).get(image, {})
        # Cache in sessione: evitiamo riletture continue del JSON mentre si
        # interagisce con il canvas e si forza spesso il rerun di Streamlit.
        st.session_state[key] = [
            [[float(x), float(y)] for x, y in p["coords"]]
            for p in (entry.get("scales") or [])
            if isinstance(p, dict) and p.get("type") == "polygon" and len(p.get("coords", [])) >= 3
        ]
    return st.session_state[key]


# ================================================================ UI

st.title("Labeling assistito da SAM")

individuals = list_individuals()
if not individuals:
    st.info("Nessun individuo trovato in data/raw/")
    st.stop()

sam, predictor, auto_gen = load_sam()
if sam is None:
    st.error(
        f"Checkpoint SAM non trovato: `{Path(SAM_CHECKPOINT).resolve()}`\n\n"
        f"```\ncurl -L -o {SAM_CHECKPOINT} "
        f"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth\n```"
    )
    st.stop()

with st.sidebar:
    individual = st.selectbox("Individuo", individuals, key="sel_individual")
    images = list_images(individual)
    if not images:
        st.warning("Nessuna immagine per questo individuo.")
        st.stop()

    # UNICA FONTE DI VERITA': l'indice in session_state.
    # La selectbox NON scrive direttamente sull'indice (era il bug che
    # annullava Prev/Next: la selectbox manteneva il suo stato interno e
    # sovrascriveva l'indice appena cambiato). Qui la selectbox usa un
    # callback esplicito, e la sua key cambia con l'indice cosi' resta
    # sempre sincronizzata con i bottoni.
    idx_key = f"img_idx_{individual}"
    if idx_key not in st.session_state:
        st.session_state[idx_key] = 0
    idx = max(0, min(st.session_state[idx_key], len(images) - 1))
    st.session_state[idx_key] = idx

    prev_col, next_col = st.columns(2)
    with prev_col:
        if st.button("⬅️ Prev", use_container_width=True, disabled=idx == 0, key=f"prev_{individual}_{idx}"):
            st.session_state[idx_key] = idx - 1
            st.rerun()
    with next_col:
        if st.button("Next ➡️", use_container_width=True, disabled=idx >= len(images) - 1, key=f"next_{individual}_{idx}"):
            st.session_state[idx_key] = idx + 1
            st.rerun()

    picked = st.selectbox("Vai a immagine", images, index=idx, key=f"imgsel_{individual}_{idx}")
    if picked != images[idx]:
        st.session_state[idx_key] = images.index(picked)
        st.rerun()

    st.caption(f"Immagine {idx + 1} di {len(images)}")

    image_name = images[idx]

    st.divider()
    zoom = st.slider("Zoom", 0.2, 2.0, 1.0, 0.1, key=f"zoom_{individual}_{image_name}")

    st.divider()
    mode = st.radio(
        "Modalita'",
        ["SAM (click singolo)", "SAM (zona rettangolare)", "Manuale (poligono)", "Elimina (click su figura)"],
        key="sel_mode",
    )
    st.caption({
        "SAM (click singolo)": "Click su una scaglia: SAM la segmenta al volo.",
        "SAM (zona rettangolare)": "Disegna un rettangolo: SAM segmenta tutto quello che trova dentro.",
        "Manuale (poligono)": "Click su ogni vertice, poi 'Chiudi poligono'.",
        "Elimina (click su figura)": "Click DENTRO una forma per eliminarla.",
    }[mode])

    st.divider()
    st.subheader("Parametri SAM")
    min_area_frac = st.number_input("Area minima (frazione immagine)", value=0.00005, format="%.5f")
    max_area_frac = st.number_input("Area massima (frazione immagine)", value=0.02, format="%.3f")
    overlap_threshold = st.slider("Soglia scarto per sovrapposizione", 0.0, 1.0, 0.2, 0.05)
    tile_size = st.number_input("Dimensione tile (px)", value=512, step=64)
    tile_overlap = st.number_input("Sovrapposizione tile (px)", value=150, step=25)

# ---------------------------------------------------------------- immagine

img_path = RAW_ROOT / individual / image_name
mtime = img_path.stat().st_mtime
pil_image = load_image_cached(str(img_path), mtime)
img_w, img_h = pil_image.size

shapes = get_shapes(individual, image_name)

version_key = f"canvas_v_{individual}_{image_name}"
st.session_state.setdefault(version_key, 0)
pending_key = f"pending_{individual}_{image_name}"
st.session_state.setdefault(pending_key, [])
pending_manual = st.session_state[pending_key]

canvas_w = int(img_w * zoom)
canvas_h = int(img_h * zoom)
if canvas_w > VIEWPORT_W or canvas_h > VIEWPORT_H:
    # Ridimensionamento solo per visualizzazione: i punti/forme restano
    # sempre convertiti su coordinate immagine originali via display_scale.
    fit = min(VIEWPORT_W / canvas_w, VIEWPORT_H / canvas_h)
    canvas_w, canvas_h = int(canvas_w * fit), int(canvas_h * fit)
display_scale = canvas_w / img_w

preview = resized_bg_cached(str(img_path), mtime, canvas_w, canvas_h).copy()
draw = ImageDraw.Draw(preview, "RGBA")
for poly in shapes:
    draw.polygon([(x * display_scale, y * display_scale) for x, y in poly],
                 outline=(0, 255, 136, 255), width=2, fill=(0, 255, 136, 40))

if mode == "Manuale (poligono)" and pending_manual:
    pts = [(x * display_scale, y * display_scale) for x, y in pending_manual]
    for p in pts:
        draw.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], fill=(255, 140, 0, 255))
    if len(pts) > 1:
        draw.line(pts, fill=(255, 140, 0, 255), width=2)

drawing_mode = "rect" if mode == "SAM (zona rettangolare)" else "point"

left_col, right_col = st.columns([4, 2])

with left_col:
    canvas_result = st_canvas(
        fill_color="rgba(255, 140, 0, 0.2)",
        stroke_width=2,
        stroke_color="#ff8c00",
        background_image=preview,
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode=drawing_mode,
        point_display_radius=POINT_RADIUS,
        key=f"canvas_{individual}_{image_name}_{mode}_{zoom:.1f}_{st.session_state[version_key]}",
        display_toolbar=False,
    )

clicks = []
new_rect = None
if canvas_result and canvas_result.json_data:
    if mode == "SAM (zona rettangolare)":
        new_rect = parse_last_rect_from_canvas(canvas_result.json_data, display_scale)
    else:
        for obj in canvas_result.json_data.get("objects", []):
            cx = float(obj.get("left", 0)) + float(obj.get("radius", POINT_RADIUS))
            cy = float(obj.get("top", 0)) + float(obj.get("radius", POINT_RADIUS))
            clicks.append((cx / display_scale, cy / display_scale))

with right_col:
    st.metric("Scaglie su questa immagine", len(shapes))

    # ------------------------------------------------ zona rettangolare
    if mode == "SAM (zona rettangolare)" and new_rect:
        rx1, ry1, rx2, ry2 = [int(v) for v in new_rect]
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(img_w, rx2), min(img_h, ry2)
        rw, rh = rx2 - rx1, ry2 - ry1
        st.info(f"Zona: {rw}x{rh}px")

        if rw >= 5 and rh >= 5 and st.button("🪄 Segmenta questa zona", type="primary", use_container_width=True):
            img_arr = load_image_array(str(img_path), mtime)
            with st.spinner("SAM sta segmentando la zona..."):
                crop_masks = auto_gen.generate(img_arr[ry1:ry2, rx1:rx2])
                total_area = img_w * img_h
                min_a, max_a = min_area_frac * total_area, max_area_frac * total_area
                existing = rasterize_shapes(shapes, img_w, img_h)
                n_added = n_skip = 0
                for m in crop_masks:
                    if not (min_a <= m["area"] <= max_a):
                        continue
                    pc = mask_to_polygon(m["segmentation"])
                    if not pc:
                        continue
                    pf = [[x + rx1, y + ry1] for x, y in pc]
                    ov, bbox, lm = overlap_fraction_fast(pf, existing, img_w, img_h)
                    if lm.size == 0:
                        continue
                    if ov > overlap_threshold:
                        n_skip += 1
                        continue
                    shapes.append(pf)
                    bx1, by1, bx2, by2 = bbox
                    existing[by1:by2, bx1:bx2] |= lm
                    n_added += 1
            persist_shapes(individual, image_name, shapes)
            st.session_state[version_key] += 1
            st.success(f"Aggiunte {n_added} scaglie ({n_skip} scartate: gia' labellate).")
            st.rerun()

    # ------------------------------------------------ click
    if clicks:
        if mode == "SAM (click singolo)":
            img_arr = load_image_array(str(img_path), mtime)
            emb_key = "sam_embedded_image"
            if st.session_state.get(emb_key) != str(img_path):
                with st.spinner("Preparo SAM per questa immagine..."):
                    predictor.set_image(img_arr)
                st.session_state[emb_key] = str(img_path)

            n_ok = 0
            for cx, cy in clicks:
                masks, scores, _ = predictor.predict(
                    point_coords=np.array([[cx, cy]]), point_labels=np.array([1]), multimask_output=True,
                )
                poly = mask_to_polygon(masks[np.argmax(scores)])
                if poly:
                    shapes.append(poly)
                    n_ok += 1
            if n_ok:
                persist_shapes(individual, image_name, shapes)
            st.session_state[version_key] += 1
            st.rerun()

        elif mode == "Manuale (poligono)":
            pending_manual.extend(clicks)
            st.session_state[version_key] += 1
            st.rerun()

        elif mode == "Elimina (click su figura)":
            n_removed = 0
            for cx, cy in clicks:
                hits = [(i, polygon_area(s)) for i, s in enumerate(shapes) if point_in_polygon(cx, cy, s)]
                if hits:
                    shapes.pop(min(hits, key=lambda t: t[1])[0])
                    n_removed += 1
            if n_removed:
                persist_shapes(individual, image_name, shapes)
            st.session_state[version_key] += 1
            st.rerun()

    # ------------------------------------------------ controlli manuale
    if mode == "Manuale (poligono)":
        st.metric("Vertici in corso", len(pending_manual))
        if st.button("✅ Chiudi poligono", type="primary", use_container_width=True,
                     disabled=len(pending_manual) < 3):
            shapes.append(list(pending_manual))
            persist_shapes(individual, image_name, shapes)
            st.session_state[pending_key] = []
            st.session_state[version_key] += 1
            st.rerun()
        if st.button("Azzera vertici", use_container_width=True, disabled=not pending_manual):
            st.session_state[pending_key] = []
            st.session_state[version_key] += 1
            st.rerun()
        st.divider()

    # ------------------------------------------------ suggerisci tutte
    if st.button("🪄 Suggerisci tutte (SAM a tile)", type="primary", use_container_width=True):
        img_arr = load_image_array(str(img_path), mtime)
        total_area = img_w * img_h
        min_a, max_a = min_area_frac * total_area, max_area_frac * total_area
        tiles = compute_tile_grid(img_w, img_h, tile_size, tile_overlap)

        bar = st.progress(0.0)
        status = st.empty()
        existing = rasterize_shapes(shapes, img_w, img_h)
        n_found = n_added = n_skip = 0

        for ti, (tx1, ty1, tx2, ty2) in enumerate(tiles):
            status.text(f"Tile {ti + 1}/{len(tiles)}")
            bar.progress((ti + 1) / len(tiles))
            tile_masks = auto_gen.generate(img_arr[ty1:ty2, tx1:tx2])
            n_found += len(tile_masks)

            for m in tile_masks:
                if not (min_a <= m["area"] <= max_a):
                    continue
                pt = mask_to_polygon(m["segmentation"])
                if not pt:
                    continue
                pf = [[x + tx1, y + ty1] for x, y in pt]
                ov, bbox, lm = overlap_fraction_fast(pf, existing, img_w, img_h)
                if lm.size == 0:
                    continue
                if ov > overlap_threshold:
                    n_skip += 1
                    continue
                shapes.append(pf)
                bx1, by1, bx2, by2 = bbox
                existing[by1:by2, bx1:bx2] |= lm
                n_added += 1

        status.empty()
        bar.empty()
        persist_shapes(individual, image_name, shapes)
        st.session_state[version_key] += 1
        st.success(f"Aggiunte {n_added} scaglie su {n_found} regioni ({len(tiles)} tile, {n_skip} scartate).")
        st.rerun()

    if st.button("🗑 Svuota tutte", use_container_width=True, disabled=not shapes):
        st.session_state[state_key(individual, image_name)] = []
        persist_shapes(individual, image_name, [])
        st.session_state[version_key] += 1
        st.rerun()

st.caption("Salvataggio automatico in data/labels/{individuo}.json ad ogni modifica.")