"""
GUI veloce: data una cartella con dentro "images/" e "labels/" (formato
YOLO-seg, coordinate normalizzate 0-1), mostra ogni immagine con la sua
label sovrapposta, con Prev/Next per scorrerle tutte.

Gestisce sia struttura piatta (images/*.jpg direttamente) sia con
sottocartelle train/val (images/train/*.jpg, images/val/*.jpg).

Uso:
  streamlit run scripts/browse_yolo_dataset.py
"""

from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="Browser dataset YOLO", layout="wide")
st.title("Browser dataset YOLO — immagine + label sovrapposta")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images_and_labels(root: Path, split: str | None) -> list[tuple[Path, Path | None]]:
    """Ritorna lista di (path_immagine, path_label_o_None), ordinata."""
    images_dir = root / "images" / split if split else root / "images"
    labels_dir = root / "labels" / split if split else root / "labels"

    if not images_dir.exists():
        return []

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    pairs = []
    for img_path in images:
        label_path = labels_dir / f"{img_path.stem}.txt"
        pairs.append((img_path, label_path if label_path.exists() else None))
    return pairs


def parse_yolo_label(label_path: Path, img_w: int, img_h: int) -> list[list[list[float]]]:
    """Legge un file label YOLO-seg (coordinate normalizzate 0-1),
    ritorna lista di poligoni in PIXEL ASSOLUTI."""
    if not label_path or not label_path.exists():
        return []

    polygons = []
    for line in label_path.read_text(encoding="utf-8").strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 7:  # class_id + almeno 3 punti (6 valori)
            continue
        coords_norm = [float(v) for v in parts[1:]]
        if len(coords_norm) % 2 != 0:
            continue
        points = []
        for i in range(0, len(coords_norm), 2):
            x = coords_norm[i] * img_w
            y = coords_norm[i + 1] * img_h
            points.append([x, y])
        if len(points) >= 3:
            polygons.append(points)
    return polygons


with st.sidebar:
    root_str = st.text_input("Cartella dataset (con dentro images/ e labels/)", value="data/yolo_dataset/prime_patches")
    root = Path(root_str)

    if not root.exists():
        st.error(f"Cartella non trovata: {root.resolve()}")
        st.stop()

    images_dir = root / "images"
    has_splits = (images_dir / "train").exists() or (images_dir / "val").exists()

    split = None
    if has_splits:
        available_splits = [s for s in ["train", "val"] if (images_dir / s).exists()]
        split = st.selectbox("Split", available_splits)

    pairs = find_images_and_labels(root, split)
    if not pairs:
        st.warning("Nessuna immagine trovata in questa cartella.")
        st.stop()

    st.metric("Immagini trovate", len(pairs))
    n_with_label = sum(1 for _, lbl in pairs if lbl is not None)
    st.caption(f"Con label: {n_with_label} · Senza label: {len(pairs) - n_with_label}")

    st.divider()
    show_fill = st.checkbox("Riempimento semi-trasparente", value=True)
    only_missing = st.checkbox("Mostra solo immagini SENZA label", value=False)

if only_missing:
    filtered_pairs = [(img, lbl) for img, lbl in pairs if lbl is None]
    if not filtered_pairs:
        st.success("Tutte le immagini hanno una label corrispondente.")
        st.stop()
    pairs = filtered_pairs

idx_key = f"browse_idx_{root_str}_{split}_{only_missing}"
if idx_key not in st.session_state:
    st.session_state[idx_key] = 0
idx = max(0, min(st.session_state[idx_key], len(pairs) - 1))
st.session_state[idx_key] = idx

img_path, label_path = pairs[idx]

nav_col1, nav_col2, nav_col3 = st.columns([1, 3, 1])
with nav_col1:
    if st.button("⬅️ Precedente", use_container_width=True, disabled=idx == 0):
        st.session_state[idx_key] = idx - 1
        st.rerun()
with nav_col3:
    if st.button("Successiva ➡️", use_container_width=True, disabled=idx >= len(pairs) - 1):
        st.session_state[idx_key] = idx + 1
        st.rerun()
with nav_col2:
    st.caption(f"Immagine {idx + 1} di {len(pairs)} — {img_path.name}")
    jump_to = st.number_input("Vai a #", min_value=1, max_value=len(pairs), value=idx + 1, label_visibility="collapsed")
    if jump_to - 1 != idx:
        st.session_state[idx_key] = jump_to - 1
        st.rerun()

image = Image.open(img_path).convert("RGB")
img_w, img_h = image.size

polygons = parse_yolo_label(label_path, img_w, img_h)

overlay = image.copy()
draw = ImageDraw.Draw(overlay, "RGBA")
for poly in polygons:
    pts = [(x, y) for x, y in poly]
    fill = (0, 255, 136, 60) if show_fill else None
    draw.polygon(pts, outline=(0, 255, 136, 255), width=2, fill=fill)

left_col, right_col = st.columns([4, 1])
with left_col:
    st.image(overlay, use_container_width=True)
with right_col:
    st.metric("Poligoni in questa label", len(polygons))
    if label_path is None:
        st.warning("Nessun file label trovato per questa immagine.")
    else:
        st.caption(f"Label: {label_path.name}")
    st.caption(f"Dimensione immagine: {img_w}x{img_h}px")
