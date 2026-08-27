# Crocodile Scale Segmentation Pipeline

A computer vision pipeline for detecting and segmenting individual crocodile scales to support traceability and anti-counterfeiting workflows in luxury leather goods.

## What This Project Does

Luxury leather products can be identified by the distinctive spatial arrangement and shape of their natural crocodile scales. This project turns that visual structure into machine-readable evidence: it detects individual scales and produces instance segmentation masks that can support inspection, traceability, and authenticity analysis. The repository contains the processing and training tools; customer images, labels, and trained weights remain outside the public repository.

The workflow starts with assisted annotation. Segment Anything (SAM) proposes scale boundaries from clicks, rectangular regions, or automatic tiled processing, while manual polygon editing remains available for corrections. Labels can be cleaned and rasterized, then used to create training data from real images. Synthetic copy-paste images and externally generated pattern images provide additional variation when the number of annotated examples is limited.

The final model is YOLO11-seg from Ultralytics. For very large source photographs, inference uses multiple tile sizes plus extra tiles centered on tile junctions. Predictions are mapped back to the full image, geometrically clipped when masks overlap, filtered for low local contrast, and saved as an annotated JPEG together with a JSON summary.

## Example Result

<!-- TODO: add example image -->
![result](docs/esempio_risultato.png)

## Installation

Run the commands below from the repository root. A virtual environment is recommended.

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies listed in `requirements.txt`:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file covers Streamlit, the drawable canvas component, Pillow, NumPy, OpenCV, PyTorch, torchvision, Segment Anything, and Ultralytics. The polygon clipping code also imports Shapely, so install it explicitly if it is not already available:

```bash
python -m pip install shapely
```

Download the SAM ViT-B checkpoint expected by the labeling application:

```bash
curl -L -o models/sam_vit_b.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

The SAM application uses CPU by default. YOLO model files such as `yolo11n-seg.pt` or `yolo11s-seg.pt` are downloaded by Ultralytics when selected and are not required to be committed to the public repository.

## Pipeline - Step-by-Step Usage

### 1. Prepare the source data

Place original images under `data/raw/<individual>/` and create `data/labels/<individual>.json`. The JSON keys must match the image filenames exactly. Source polygons use absolute pixel coordinates and must contain at least three points.

Expected layout:

```text
data/
  raw/
    <individual>/
      image_001.bmp
  labels/
    <individual>.json
```

### 2. Assisted labeling with SAM

Launch the Streamlit application:

```bash
streamlit run scripts/sam_assisted_labeling.py
```

The app supports single-click SAM segmentation, rectangular-zone automatic segmentation, manual polygons, deletion by click, and a `Suggest all` tiled SAM mode. It automatically saves the current image labels to `data/labels/<individual>.json`. Its main interactive defaults are a 512 px tile, 150 px tile overlap, a minimum area fraction of `0.00005`, a maximum area fraction of `0.02`, and an overlap rejection threshold of `0.2`.

### 3. Inspect labels before processing

For a quick visual check of one image and one JSON file, run:

```bash
streamlit run scripts/quick_label_check.py
```

This accepts the standard project JSON format as well as an image entry containing `scales`, a direct list of shape dictionaries, or a direct list of polygons. It draws the recognized polygons over the uploaded image and can number them.

### 4. Clean overlapping polygons

Generate review previews without changing the source JSON:

```bash
python scripts/clean_overlapping_scales.py
```

Process one individual with a five-pixel separation margin:

```bash
python scripts/clean_overlapping_scales.py --individual prime --margin 5
```

After reviewing the previews in `data/cleaning_preview/`, apply the cleaned polygons and create a `.json.bak` backup:

```bash
python scripts/clean_overlapping_scales.py --apply
```

The script rasterizes, erodes, removes residual overlaps, reconstructs polygons, and discards shapes that become too small. Without `--apply`, it only writes preview images.

### 5. Rasterize masks (optional ground-truth output)

Generate full-resolution binary PNG masks from the JSON polygons:

```bash
python scripts/rasterize_masks.py
```

For one individual only:

```bash
python scripts/rasterize_masks.py --individual prime
```

White pixels represent the union of labeled scales and black pixels represent the background or gaps. These masks are auxiliary outputs; the YOLO training flow below uses YOLO-seg polygon labels generated for the training images.

### 6. Add synthetic data (choose one or both sources)

To import externally generated pattern images, where each image has a matching `{name}_labels.json`, run:

```bash
python scripts/import_synthetic_patterns.py --source "data/sintetiche/prova 1/generated_patterns" --output-name pattern_synth
```

The script copies supported image files into `data/raw/pattern_synth/` and combines the label JSON files into `data/labels/pattern_synth.json`.

To generate copy-paste composites from real labeled scales, run:

```bash
python scripts/generate_copypaste_dataset.py --individual prime --n-images 100
```

Useful controls include `--canvas-size 768`, `--scales-per-row 12`, `--packing 0.55`, `--max-overlap 0.10`, `--hole-prob 0.04`, `--blob-coverage 0.88`, `--max-rotation 8.0`, `--feather 5.0`, `--output-name <name>`, and `--val-frac 0.15`. The script writes a YOLO dataset directly under `data/yolo_dataset/<name>/` and also writes a raw-image and project-label representation under `data/raw/<name>/` and `data/labels/<name>.json`.

### 7. Create labeled multi-scale patches

For a single labeled source, extract patches, clip polygons to patch boundaries with Shapely, resize them, and write YOLO-seg labels:

```bash
python scripts/extract_labeled_patches_yolo.py --individual prime --scales 224 400 --output-size 224
```

A fuller example with custom splitting and a patch limit is:

```bash
python scripts/extract_labeled_patches_yolo.py --individual prime --scales 224 400 600 --overlap 0.25 --output-size 224 --val-frac 0.15 --output-name prime_patches_labeled --max-patches 5000
```

Its principal options are `--individual`, `--scales`, `--overlap`, `--output-size`, `--val-frac`, `--output-name`, and `--max-patches`. It skips images with fewer than three valid polygons and writes `data.yaml` in the resulting dataset directory.

If patches already exist in `data/patches/<run-folder>/`, convert them instead:

```bash
python scripts/prepare_yolo_from_patches.py --run-folder prime_dense128
```

Optional controls are `--output-name`, `--val-frac 0.15`, `--test-frac 0.0`, `--seed 42`, and `--overlap 0.25`. The script uses `_meta.json` when present and otherwise attempts to reconstruct metadata from `data/patch_zones/` and `data/labels/`.

### 8. Recommended complete dataset build

For real data plus one or more synthetic sources, the recommended command performs the image-level train/validation/test split before patch extraction, extracts multi-scale patches, balances each synthetic source against the real source, and writes one final YOLO dataset:

```bash
python scripts/build_full_yolo_pipeline.py --real prime --synthetic pattern_synth pattern_synth2 --scales 128 256 384 512 640 768 896 1024 --output-size 224 --output-name dataset_finale
```

Important options are `--real`, `--synthetic`, `--scales`, `--overlap 0.0`, `--output-size 224`, `--val-frac 0`, `--test-frac 0.1`, `--output-name`, `--force`, and `--auto-train`. Add `--epochs 100` to control the training launched by `--auto-train`:

```bash
python scripts/build_full_yolo_pipeline.py --real prime --synthetic pattern_synth --output-name dataset_finale --auto-train --epochs 100
```

The per-source cache is stored in `data/yolo_dataset/_per_source/`; use `--force` to rebuild it. Splitting at the original-image level prevents patches from the same photograph leaking into different evaluation splits.

### 9. Browse a generated YOLO dataset

Inspect images and their YOLO-seg polygons interactively:

```bash
streamlit run scripts/browse_yolo_dataset.py
```

The default dataset path is `data/yolo_dataset/prime_patches`. The sidebar can select another root containing `images/` and `labels/`, choose `train` or `val` when splits exist, enable translucent fills, and filter images without matching labels.

### 10. Train YOLO11-seg

Train on one dataset:

```bash
python scripts/train_yolo.py --data data/yolo_dataset/dataset_finale/data.yaml --model yolo11s-seg.pt --epochs 100 --imgsz 1024 --batch 4 --cache disk
```

Train on several datasets in one run:

```bash
python scripts/train_yolo.py --data data/yolo_dataset/prime/data.yaml data/yolo_dataset/prime_copypaste/data.yaml --model yolo11s-seg.pt --epochs 100 --imgsz 1024 --batch 4
```

The CLI options are `--data` (one or more YAML files), `--model` (`yolo11n-seg.pt`, `yolo11s-seg.pt`, `yolo11m-seg.pt`, or `yolo11l-seg.pt`), `--epochs`, `--imgsz`, `--batch`, `--cache` (`ram`, `disk`, or `none`), and `--output-name`. Results are written under `models/yolo_runs/`, with experiment metadata under `models/training_logs/`.

### 11. Evaluate on a synthetic dataset

Evaluate a trained checkpoint on the selected split:

```bash
python scripts/eval_on_synthetic.py --model models/yolo_runs/<run>/weights/best.pt --data data/yolo_dataset/<synthetic>/data.yaml --split val --imgsz 1024 --conf 0.25 --iou 0.7
```

The script reports box mAP50, box mAP50-95, precision, and recall. `--split` accepts `train` or `val`.

### 12. Run full-image, multi-scale tiled inference

Run inference on a folder of original images:

```bash
python scripts/predict_yolo_tiled.py --model models/yolo_runs/<run>/weights/best.pt --folder data/test_images --tile-sizes 512 768 --conf 0.15
```

Run it on one image instead:

```bash
python scripts/predict_yolo_tiled.py --model models/yolo_runs/<run>/weights/best.pt --image path/to/image.jpg --tile-sizes 512 768 --overlap 150 --batch 2 --output-dir data/yolo_predictions_full
```

Alternatively, use images from a generated YOLO dataset:

```bash
python scripts/predict_yolo_tiled.py --model models/yolo_runs/<run>/weights/best.pt --yolo-dataset data/yolo_dataset/dataset_finale --tile-sizes 512 768
```

Exactly one of `--image`, `--folder`, or `--yolo-dataset` is required. Other principal controls are `--tile-sizes`, `--junction-tile-size`, `--max-junction-tiles 300`, `--overlap 150`, `--conf 0.15`, `--batch 2`, `--min-local-contrast 8.0`, `--smoothing 2`, `--output-dir`, and `--jpeg-quality 75`. The `--nms-iou` argument is retained for command compatibility but is no longer used for overlap removal.

### 13. Package data for transfer (optional)

To build a separate transfer package containing scripts, raw images, labels, rasterized masks, patch zones, and requirements:

```bash
python scripts/build_transfer_package.py
```

This command recreates `transfer_package/` and excludes caches, patch outputs, embeddings, test outputs, previews, and model checkpoints according to the script's configured inclusion and exclusion lists.

## Project Structure

```text
.
├── data/
│   ├── raw/                  # Original images grouped by individual
│   ├── labels/               # Source JSON polygons in absolute pixels
│   ├── masks/                # Optional rasterized masks
│   ├── patch_zones/          # Optional patch-zone metadata
│   ├── patches/              # Existing patch runs, when used
│   └── yolo_dataset/         # Generated YOLO images, labels, and YAML files
├── models/
│   ├── training_logs/        # JSON experiment summaries
│   └── yolo_runs/            # Ultralytics training outputs and weights
├── scripts/                  # Labeling, data preparation, training, and inference tools
├── requirements.txt
└── README.md
```

Customer data, raw images, labels, checkpoints, generated datasets, and trained models are intentionally not included in the public repository for confidentiality reasons. Keep them in the local working directories described above.

## License and Contact

Add the applicable license before publishing or redistributing this repository. For questions about the pipeline, contact [Nome].
