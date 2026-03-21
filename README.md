# SL-HOI
Implementation of Streamlined Open-Vocabulary Human-Object Interaction Detection (CVPR 2026)

## Overview

In this paper, we present SL-HOI, a streamlined one-stage framework for open-vocabulary HOI detection built upon the DINOv3 model. We leverage the complementary strengths of DINOv3's backbone and vision head to effectively address both interactive human-object detection and open-vocabulary interaction classification tasks. Our design includes a novel two-step interaction classification process that bridges representation gaps and enhances feature utilization. Extensive experiments on two popular benchmarks demonstrate that SL-HOI achieves state-of-the-art performance in open-vocabulary HOI detection while maintaining a simple architecture with few trainable parameters.

<!-- TODO -->

## Installation

### Requirements

- Python 3.10
- PyTorch 2.5.1
- CUDA ≥ 12.1
- transformers
- accelerate
- deepspeed

A `requirements.txt` file will be provided later.

### Setup

```bash
git clone https://github.com/your-repo/SL-HOI.git
cd SL-HOI
pip install -r requirements.txt
```

## Data Preparation

### HICO-DET

HICO-DET dataset preparation follows [GEN-VLKT](https://github.com/YueLiao/GEN-VLKT). Please refer to their documentation for download and setup instructions.

```
hico_20160224_det
 |─ images
 |   |─ train2015
 |   |─ test2015
 |─ annotations
 |   |─ trainval_hico.json
 |   |─ test_hico.json
 |   |─ corre_hico.npy
```

### SWIG-HOI

SWIG-HOI dataset preparation follows [THID](https://github.com/scwangdyd/promting_hoi). Please refer to their documentation for download and setup instructions.

```
swig_hoi
 |─ images_512
 |─ annotations
 |   |─ swig_train_1000.json
 |   |─ swig_val_1000.json
 |   |─ swig_trainval_1000.json
 |   |─ swig_test_1000.json
```

## Model Weights

### Pretrained Backbones

Download the DINOv3 pretrained weights:

**Official weights:**
- DINOv3: [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

**Our converted weights (coming soon):**
- DINOv3 ViT-L/16 backbone
- DINOv3-txt (Vision+Text encoder)

### HOI Classifiers

HOI classifier weights can be generated using the provided scripts:

```bash
# Generate HICO-DET classifiers
python hico_offline_classifier.py

# Generate SWIG-HOI classifiers
python swig_offline_classifier.py
```

By default, the classifier weights will be saved in `params`

Pre-computed classifier weights will be made available for download (coming soon).

## Training

Training scripts are provided in `scripts/`:

- `scripts/swig.sh` - Training on SWIG-HOI
- `scripts/hico.sh` - Training on HICO-DET
- `scripts/hico_ov.sh` - Training on HICO-DET with zero-shot setting

Modify the following variables in the scripts to match your environment:

```bash
EXP_DIR="exps/swig"                    # Experiment output directory
DATA_DIR="/path/to/your/datasets"      # Path to dataset
DINO_DIR="/path/to/your/weights"       # Path to DINOv3 weights
```

Then run:

```bash
bash scripts/swig.sh
```

## Evaluation

Evaluation scripts are provided in `scripts/`:

- `scripts/swig_eval.sh` - Evaluate on SWIG-HOI
- `scripts/hico_eval.sh` - Evaluate on HICO-DET
- `scripts/hico_ov_eval.sh` - Evaluate on HICO-DET with zero-shot setting

Place the provided checkpoints in the `pretrained` folder. Modify only `DATA_DIR` in the evaluation scripts to point to your dataset, then run:

```bash
bash scripts/swig_eval.sh
```

### Performance

| Dataset | Setting | Unseen | Rare | Non-rare/ Seen | Full | Checkpoint |
|---------|---------|--------|------|----------------|------|------------|
| SWIG-HOI | - | 19.04 | 24.69 | 30.62 | 24.67 | (coming soon) |
| HICO-DET | Default | - | 47.71 | 44.25 | 45.05 | (coming soon) |
| HICO-DET | Zero-shot | 40.53 | - | 42.99 | 42.49 | (coming soon) |

## Citation

```bibtex
@inproceedings{slhoi2026,
  title={Streamlined Open-Vocabulary Human-Object Interaction Detection},
  author={Chang Sun and Dongliang Liao and Changxing Ding},
  booktitle={CVPR},
  year={2026}
}
```

## Acknowledgments

This code builds upon [QPIC](https://github.com/hitachi-rd-cv/qpic), [GEN-VLKT](https://github.com/YueLiao/GEN-VLKT), [THID](https://github.com/scwangdyd/promting_hoi), and [DINOv3](https://github.com/facebookresearch/dinov3). We thank their authors for making their code publicly available.
