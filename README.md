# EuroSAT MS Land-Cover Classification

Sentinel-2 multispectral land-use classification using fine-tuned ResNet-18 backbones,
spectral index features, and a multi-stream late-fusion pipeline.

## Setup

```bash
conda create -n eurosat python=3.10 -y
conda activate eurosat
pip install -r requirements.txt
```

**Data:** place the EuroSAT multispectral dataset at `/Eurosat_Data/EuroSAT_MS/`
(or update `paths.ms_root` in `config.yaml`).

---

## Quick start

### 1 Generate the 80/20 split (run once)

```bash
python data/splits.py
python data/splits.py --show    # verify per-class counts
```

### 2 :Train

```bash
python train.py --modality rgb
python train.py --modality ms
```

Checkpoints are saved to `results/rgb/best.pt` and `results/ms/best.pt`.

### 3 :Test / evaluate

```bash
python test.py --modality rgb --checkpoint results/rgb/best.pt --visualize
python test.py --modality ms  --checkpoint results/ms/best.pt  --visualize
```

Outputs: `metrics.json`, `metrics.csv`, `confusion.png`, `training_curves.png`, `sample_gallery.png`.

### 4 :ML / spectral-index arms

```bash
# Extract spectral index features
python features/spectral_indices.py

# Extract fine-tuned CNN features (run after training)
python features/cnn_extractor.py --modality rgb --finetuned
python features/cnn_extractor.py --modality ms  --finetuned

# Train and evaluate all LightGBM arms
python train_ml.py

# Single arm
python train_ml.py --arm ms_cnn_indices
```

### 5 :Late-fusion pipeline

```bash
python fusion_ft/run_fusion.py
python fusion_ft/run_fusion.py --no-xgb     # skip XGBoost comparison
python fusion_ft/run_fusion.py --arm ms_fusion   # single arm
```

### 6 :Full end-to-end run

```bash
bash scripts/run_all.sh
```

---

## Common CLI flags

| Script | Key flags |
|--------|-----------|
| `train.py` | `--modality {rgb,ms}`, `--epochs`, `--batch-size`, `--backbone-lr`, `--head-lr`, `--force`, `--smoke-test` |
| `test.py`  | `--modality {rgb,ms}`, `--checkpoint`, `--batch-size`, `--visualize` |
| `train_ml.py` | `--arm {indices_only,rgb_cnn,ms_cnn,ms_cnn_indices,all}`, `--force`, `--eval-only`, `--atmos-ablation` |
| `fusion_ft/run_fusion.py` | `--arm`, `--force`, `--no-xgb`, `--embed-only`, `--hc-only` |

---

## Project layout

```
MS_Project_v2/
├── config.yaml              # all settings — edit here
├── settings.py              # config loader (read-only)
├── train.py                 # fine-tune backbone + head
├── test.py                  # evaluate checkpoint on test set
├── train_ml.py              # LightGBM arms
│
├── data/
│   ├── splits.py            # 80/20 stratified split builder
│   ├── dataset.py           # EuroSATDataset: rgb (3-ch) or ms (10-ch)
│   └── transforms.py        # SentinelTransform + AugmentTransform
│
├── models/
│   ├── backbone.py          # pretrained ResNet-18 loader (10-band conv1 for MS)
│   └── classifier.py        # TwoLayerFC head + FinetuneModel
│
├── features/
│   ├── spectral_indices.py  # 32-d spectral index features
│   └── cnn_extractor.py     # 512-d CNN embedding extraction
│
├── evaluation/
│   └── metrics.py           # accuracy, macro-F1, per-class, McNemar
│
├── visualization/
│   ├── confusion.py
│   ├── curves.py
│   ├── gallery.py           # sample test patches with predicted labels
│   └── shap_viz.py
│
├── fusion_ft/               # multi-stream late-fusion pipeline
│   ├── embeddings.py
│   ├── handcrafted.py       # spectral (32-d) + texture (22-d) features
│   ├── fusion_arms.py       # 11 arm definitions + feature loading
│   └── run_fusion.py        # training, evaluation, SHAP, summary
│
├── scripts/
│   └── run_all.sh           # end-to-end reproduction script
└── requirements.txt
```

---

## Design notes

**Split:** 80% train / 20% test (`data/splits.npz`).  The validation set
is created from the 80% train indices *at training time*; it is not stored so the
test set remains completely held out.

**MS band handling:** EuroSAT GeoTIFFs store B8A last (index 12 instead of 8).
The dataset reorders to SSL4EO-S12 canonical order, then drops the three atmospheric
bands (B01, B09, B10) that carry no land-cover signal.  The result is a 10-channel
input: B02 B03 B04 B05 B06 B07 B08 B8A B11 B12.

**MS backbone:** starts from SENTINEL2_ALL_MOCO (13-channel pretrained). The first
conv layer is replaced with a 10-channel version whose weights are sliced from the
pretrained 13-channel filters only the land-band slices are kept.  This reuses
pretrained spectral knowledge for every band the model actually sees.

**Discriminative learning rates:** backbone at 1e-4, head at 1e-3.  The pretrained
backbone needs gentle nudging; the randomly initialised head needs faster convergence.
