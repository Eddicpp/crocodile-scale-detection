"""
Verifica rapida: carichi un'immagine + il suo file JSON di label, vedi
subito i poligoni sovrapposti per controllare che corrispondano.

Accetta diversi formati JSON senza bisogno di adattarli a mano:
  - {"nome_immagine.ext": {"scales": [{"type":"polygon","coords":[[x,y],...]}]}}
    (formato standard del progetto, data/labels/{individuo}.json)
  - {"scales": [...]} (solo l'entry di una immagine, senza il nome file attorno)
  - [{"type":"polygon","coords":[...]}, ...] (lista diretta di forme)
  - [[[x,y],...], [[x,y],...]] (lista diretta di poligoni, senza wrapper)

Uso:
  streamlit run scripts/quick_label_check.py
"""

import json

import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="Verifica rapida label", layout="wide")
st.title("Verifica rapida: immagine + label")
st.caption("Carica un'immagine e il suo JSON di label per vedere subito se corrispondono.")


def extract_polygons(data, image_name: str | None = None) -> list:
    """Prova a estrarre una lista di poligoni [[x,y],...] da qualunque
    forma ragionevole di JSON, senza richiedere che l'utente adatti il
    formato a mano."""

    # caso: dict con chiavi = nomi immagine (formato standard del progetto)
    if isinstance(data, dict) and "scales" not in data:
        entry = None
        if image_name and image_name in data:
            entry = data[image_name]
        elif len(data) == 1:
            entry = next(iter(data.values()))
        else:
            # nessuna corrispondenza esatta col nome file: prova il primo
            entry = next(iter(data.values())) if data else None
        if entry is not None:
            return extract_polygons(entry, image_name)
        return []

    # caso: {"scales": [...]}
    if isinstance(data, dict) and "scales" in data:
        return extract_polygons(data["scales"], image_name)

    # caso: lista di shape-dict {"type":"polygon","coords":[...]}
    if isinstance(data, list) and data and isinstance(data[0], dict) and "coords" in data[0]:
        return [
            [[float(x), float(y)] for x, y in s["coords"]]
            for s in data if isinstance(s, dict) and len(s.get("coords", [])) >= 3
        ]

    # caso: lista diretta di poligoni [[[x,y],...], ...]
    if isinstance(data, list) and data and isinstance(data[0], list):
        return [[[float(x), float(y)] for x, y in poly] for poly in data if len(poly) >= 3]

    return []


left_col, right_col = st.columns(2)

with left_col:
    img_file = st.file_uploader("Immagine", type=["png", "jpg", "jpeg", "bmp"])

with right_col:
    json_file = st.file_uploader("File JSON delle label", type=["json"])

if img_file and json_file:
    image = Image.open(img_file).convert("RGB")

    try:
        raw_text = json_file.read().decode("utf-8")
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        st.error(f"JSON non valido: {e}")
        st.stop()

    polygons = extract_polygons(data, image_name=img_file.name)

    st.divider()
    col_view, col_info = st.columns([3, 1])

    with col_info:
        st.metric("Poligoni trovati nel JSON", len(polygons))
        show_fill = st.checkbox("Riempimento semi-trasparente", value=True)
        show_numbers = st.checkbox("Numera le scaglie", value=False)
        opacity = st.slider("Opacita' riempimento", 0, 255, 60) if show_fill else 0

        if len(polygons) == 0:
            st.warning(
                "Nessun poligono riconosciuto nel JSON. Controlla che il formato sia uno di quelli supportati "
                "(vedi docstring dello script) o incolla qui sotto un frammento per capire la struttura."
            )
            with st.expander("Anteprima struttura JSON caricato"):
                st.json(data if not isinstance(data, dict) or len(str(data)) < 3000 else {"...": "troppo grande da mostrare intero"})

    with col_view:
        overlay = image.copy()
        draw = ImageDraw.Draw(overlay, "RGBA")
        for i, poly in enumerate(polygons):
            pts = [(x, y) for x, y in poly]
            fill = (0, 255, 136, opacity) if show_fill else None
            draw.polygon(pts, outline=(0, 255, 136, 255), width=2, fill=fill)
            if show_numbers:
                cx = sum(p[0] for p in pts) / len(pts)
                cy = sum(p[1] for p in pts) / len(pts)
                draw.text((cx, cy), str(i + 1), fill=(255, 140, 0, 255))

        st.image(overlay, use_container_width=True, caption=f"{img_file.name} — {len(polygons)} poligoni sovrapposti")

elif img_file and not json_file:
    st.image(Image.open(img_file), use_container_width=True, caption=f"{img_file.name} (carica anche il JSON per vedere le label)")
    st.info("Carica anche il file JSON per sovrapporre i poligoni.")

else:
    st.info("Carica un'immagine e il file JSON di label corrispondente per iniziare.")