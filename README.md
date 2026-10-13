
# Underwater Marine Object Detection Using a Lightweight Hybrid CNN-Transformer Architecture

BSc Computer Science final-year project, Keele University (module CSC-30014).

This project investigates whether adding multi-scale convolution, attention and a lightweight transformer block to **YOLOv8n** improves underwater object detection. Underwater images suffer from turbidity, low visibility, motion blur and cluttered backgrounds, which makes detection harder, especially on resource-constrained platforms such as underwater cameras and ROVs.

The work has three parts:

1. Build a unified, cleaned marine dataset (**Hybrid Marine COCO**) from the DUO and RUOD datasets.
2. Train a **YOLOv8n baseline** on it.
3. Implement and train **custom hybrid architectures** (MSRCB, CBAM, LiteTransformerBlock) inside the Ultralytics YOLOv8 framework and compare them to the baseline.

## Key result

The baseline won. The extra architectural complexity did not improve detection on this dataset.

| Model | Precision | Recall | mAP@0.50 | mAP@0.50:0.95 | Training time |
|---|---|---|---|---|---|
| YOLOv8n baseline | 0.648 | 0.565 | **0.569** | **0.378** | 1.15 h |
| YOLOv8 + MSRCB + CBAM (run 1) | 0.615 | 0.509 | 0.517 | 0.322 | 1.16 h |
| YOLOv8 + MSRCB + CBAM (run 2) | 0.595 | 0.520 | 0.529 | 0.333 | 1.41 h |
| YOLOv8 + MSRCB + CBAM + LiteTransformerBlock | 0.604 | 0.477 | 0.473 | 0.287 | 2.87 h |

All models trained for 150 epochs at 640 px on an NVIDIA RTX 2080 (the transformer model used batch size 4 instead of 16). Likely reasons for the result are class imbalance (e.g. *echinus* dominates while *shark* and *sea anemone* are rare), a small final dataset, and the smaller batch size needed by the transformer model. See the report for full analysis.

## Repository contents

> The report notes that early experiment file and folder names did not always match the architecture inside them. Models are described here by their verified components.

### Dataset configuration
| File | Purpose |
|---|---|
| `marine_dataset.yaml` | Ultralytics dataset config for Hybrid Marine COCO (12 classes; train/val/test paths). |

### Dataset preparation pipeline (run in this order)
| Step | File | What it does |
|---|---|---|
| 1 | `merge_marine_coco.py` | Merges DUO and RUOD into one COCO-style dataset, maps labels to a shared class set, and re-splits 70/20/10 (train/val/test). |
| 2 | `validate_coco.py` | Checks annotation integrity, image availability, category IDs and bounding boxes. |
| 3 | `visual_validate.py` | Loads the data into FiftyOne for visual inspection of labels. |
| 4 | `remove_dolphin_class.py` | Removes the dolphin class and reindexes categories, images and annotations. |
| 5 | `export_audit_sheets.py` | Exports sampled annotated images per class for manual review. |
| 6 | `make_audit_csv_from_folders.py` | Builds CSV audit sheets for selected classes. |
| 7 | `label_audit_interactive.py` | Interactive label audit (correct / wrong / unsure / skip). |
| 8 | `class_distribution_clean.py`, `class_distribution_pruned.py`, `class_distribution_final.py` | Measure class distribution at each cleaning stage. |

Final dataset: 3,742 images and 17,845 annotations across 12 classes.

### Model configurations
| File | Architecture |
|---|---|
| `yolov8_msrcb_cbam.yaml` | YOLOv8 + Multi-Scale Residual Convolution Block (MSRCB) + CBAM attention. |
| `yolov8_msrcb_cbam_litetransformer.yaml` | The above + LiteTransformerBlock in the backbone. |

The custom modules are registered in the Ultralytics model parser (see the report, Figure 9).

### Training scripts
| File | Model |
|---|---|
| `train_yolov8n_baseline_reconstructed.py` | YOLOv8n baseline. |
| `train_yolov8_msrcb_cbam_run1.py` | MSRCB + CBAM, run 1. |
| `train_yolov8_msrcb_cbam_run2.py` | MSRCB + CBAM, run 2. |
| `train_yolov8_msrcb_cbam_litetransformer_reconstructed.py` | MSRCB + CBAM + LiteTransformerBlock (batch 4, 4 workers). |

Scripts marked `_reconstructed` were rebuilt after the original runs, so they reproduce the documented settings rather than being the original files.

### Training outputs
Each run provides `args.yaml` (hyperparameters), `results.csv` (per-epoch metrics) and `best.pt` (best weights).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install ultralytics fiftyone
```

The datasets are **not** included. Download them from their original sources and place them where `merge_marine_coco.py` expects:

- **DUO**: Liu et al., 2021, *A dataset and benchmark of underwater object detection for robot picking*, https://arxiv.org/abs/2106.05681
- **RUOD**: Fu et al., 2023, *Rethinking general underwater object detection*, Neurocomputing 517.

## Reproducing the experiments

```bash
# 1. Build and clean the dataset
python merge_marine_coco.py
python validate_coco.py
python remove_dolphin_class.py

# 2. Train
python train_yolov8n_baseline_reconstructed.py
python train_yolov8_msrcb_cbam_run2.py
python train_yolov8_msrcb_cbam_litetransformer_reconstructed.py
```

Edit the dataset paths in `marine_dataset.yaml` and the scripts to match your machine.

## Limitations and future work

- Strong class imbalance in the final training split.
- Only DUO and RUOD were used; Fish4Knowledge and SeaMAPD21 were left out for time.
- Little hyperparameter tuning for the hybrid models.
- Future work: more datasets, class-aware losses / resampling, ablations isolating MSRCB, CBAM and the transformer block, and wider tuning of the transformer module.

## Acknowledgements

Built on [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). Thanks to the DUO and RUOD authors for their datasets.

## Licence

Ultralytics is AGPL-3.0, so code derived from it should be released under the same licence. Add a `LICENSE` file accordingly.
