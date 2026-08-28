# Federated-Learning-Task-Arithmetic
**Author:** Mohammadreza Babaei

**Student ID:** S343333

This repository implements CIFAR-100 classification with a DINO ViT-S/16 backbone in centralized, federated, and sparse federated training settings. The main entry point is `main.py`, and the CLI is validated against `python main.py --help`.

## Overview

The project compares:

- centralized training;
- IID federated learning;
- non-IID federated learning;
- SparseSGDM training with a shared gradient mask;
- multiple mask-selection strategies.

The dataset is CIFAR-100 from `torchvision.datasets.CIFAR100`, and the model is a DINO ViT-S/16 backbone wrapped with a CIFAR-100 classifier head.

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies in `requirements.txt`:

- torch==2.13.0
- torchvision==0.28.0
- torchaudio==2.11.0
- numpy==2.5.1
- scipy==1.18.0
- matplotlib==3.11.1
- pandas==3.0.3
- tqdm==4.69.0
- scikit-learn==1.9.0
- pyyaml==6.0.3
- pillow==12.3.0

## Dataset and model

- Dataset: CIFAR-100
- Data root: `./data`
- Default data split: 90% train / 10% validation from the CIFAR-100 training set
- Validation fraction: `--val-fraction 0.1`
- Batch size: `--batch-size 32`
- Model: DINO ViT-S/16 with a CIFAR-100 classifier head
- Pretrained backbone: enabled explicitly with `--pretrained`

The repository contains the implemented dataset builders in `data/cifar100.py`, client partition logic in `data/federated.py`, and the DINO wrapper in `models/dino_vit.py`.

## CLI defaults vs final report configurations

The CLI defaults are not the same as the configurations used in the reported experiments.

CLI defaults in `main.py`:

- `--mode centralized`
- `--batch-size 32`
- `--val-fraction 0.1`
- `--num-workers 2`
- `--epochs 100`
- `--rounds 100`
- `--local-steps 4`
- `--num-clients 100`
- `--client-fraction 0.1`
- `--learning-rate 0.01`
- `--momentum 0.9`
- `--weight-decay 0.0005`
- `--scheduler-name cosine`
- `--scheduler-step-size 10`
- `--scheduler-gamma 0.1`
- `--sparsity-ratio 0.1`
- `--calibration-rounds 1`
- `--calibration-batches 1`

Report reproduction settings used in this project:

- Centralized final config: 100 epochs, LR=0.001, momentum=0.9, weight decay=0.0005, no scheduler
- IID federated baseline: `K=100`, `C=0.1`, `J=4`, 100 rounds, LR=0.01, momentum=0.9, weight decay=0.0005, batch size=32
- Non-IID experiments: `Nc={1,5,10,50}` with `J=4 -> 100 rounds`, `J=8 -> 50 rounds`, `J=16 -> 25 rounds`
- SparseSGDM main study: `sparsity ratio = 0.1`, `calibration rounds = 10`, `calibration batches = 10`, `J=4`, 100 rounds, IID
- Additional studies: sparsity ratio `{0.05, 0.10, 0.20}` and calibration rounds `{5,10}`

## Report reproduction commands

### 1) Final centralized training command

This is a configuration example for the reported centralized setting, including a pretrained DINO backbone and no scheduler. Repeat this configuration with each corresponding report seed to reproduce the saved multi-seed results.

```bash
python main.py \
  --mode centralized \
  --data-root ./data \
  --device auto \
  --seed 42 \
  --batch-size 32 \
  --val-fraction 0.1 \
  --num-workers 2 \
  --epochs 100 \
  --learning-rate 0.001 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --scheduler-name none \
  --pretrained \
  --checkpoint-path experiments/checkpoints/best_centralized.pt \
  --last-checkpoint-path experiments/checkpoints/last_centralized.pt \
  --history-path experiments/centralized_history.csv
```

The repository includes matching artifact names such as `best_centralized.pt`, `last_centralized.pt`, and `centralized_history_none.csv`. The single `--seed 42` command above is an example; repeat the same configuration with the corresponding report seeds for the multi-seed result.

### 2) IID federated baseline

The command below is a configuration example. Repeat the same configuration with each corresponding report seed to reproduce the saved multi-seed IID results.

```bash
python main.py \
  --mode federated \
  --partition iid \
  --data-root ./data \
  --device auto \
  --seed 42 \
  --batch-size 32 \
  --val-fraction 0.1 \
  --num-workers 2 \
  --num-clients 100 \
  --client-fraction 0.1 \
  --rounds 100 \
  --local-steps 4 \
  --learning-rate 0.01 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --scheduler-name cosine \
  --scheduler-step-size 10 \
  --scheduler-gamma 0.1 \
  --pretrained \
  --federated-checkpoint-path experiments/checkpoints/best_federated_iid_j4.pt \
  --last-federated-checkpoint-path experiments/checkpoints/last_federated_iid_j4.pt
```

This corresponds to the reported `K=100, C=0.1, J=4, R=100` setting.

### 3) Non-IID federated experiments

The code accepts `--partition non_iid` and requires `--classes-per-client` with one of `1, 5, 10, 50`.

For each non-IID class count, the repository contains the reported `J`/round combinations:

- `J=4` with 100 rounds
- `J=8` with 50 rounds
- `J=16` with 25 rounds

Examples:

```bash
# Nc=1, J=4, 100 rounds
python main.py \
  --mode federated \
  --partition non_iid \
  --classes-per-client 1 \
  --data-root ./data \
  --device auto \
  --seed 42 \
  --batch-size 32 \
  --val-fraction 0.1 \
  --num-workers 2 \
  --num-clients 100 \
  --client-fraction 0.1 \
  --rounds 100 \
  --local-steps 4 \
  --learning-rate 0.01 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --scheduler-name cosine \
  --scheduler-step-size 10 \
  --scheduler-gamma 0.1 \
  --pretrained \
  --federated-checkpoint-path experiments/checkpoints/best_federated_non_iid_nc1_j4.pt \
  --last-federated-checkpoint-path experiments/checkpoints/last_federated_non_iid_nc1_j4.pt
```

```bash
# Nc=5, J=8, 50 rounds
python main.py \
  --mode federated \
  --partition non_iid \
  --classes-per-client 5 \
  --data-root ./data \
  --device auto \
  --seed 42 \
  --batch-size 32 \
  --val-fraction 0.1 \
  --num-workers 2 \
  --num-clients 100 \
  --client-fraction 0.1 \
  --rounds 50 \
  --local-steps 8 \
  --learning-rate 0.01 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --scheduler-name cosine \
  --scheduler-step-size 10 \
  --scheduler-gamma 0.1 \
  --pretrained \
  --federated-checkpoint-path experiments/checkpoints/best_federated_non_iid_nc5_j8_r50.pt \
  --last-federated-checkpoint-path experiments/checkpoints/last_federated_non_iid_nc5_j8_r50.pt
```

```bash
# Nc=10, J=16, 25 rounds
python main.py \
  --mode federated \
  --partition non_iid \
  --classes-per-client 10 \
  --data-root ./data \
  --device auto \
  --seed 42 \
  --batch-size 32 \
  --val-fraction 0.1 \
  --num-workers 2 \
  --num-clients 100 \
  --client-fraction 0.1 \
  --rounds 25 \
  --local-steps 16 \
  --learning-rate 0.01 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --scheduler-name cosine \
  --scheduler-step-size 10 \
  --scheduler-gamma 0.1 \
  --pretrained \
  --federated-checkpoint-path experiments/checkpoints/best_federated_non_iid_nc10_j16_r25.pt \
  --last-federated-checkpoint-path experiments/checkpoints/last_federated_non_iid_nc10_j16_r25.pt
```

```bash
# Nc=50, J=4, 100 rounds
python main.py \
  --mode federated \
  --partition non_iid \
  --classes-per-client 50 \
  --data-root ./data \
  --device auto \
  --seed 42 \
  --batch-size 32 \
  --val-fraction 0.1 \
  --num-workers 2 \
  --num-clients 100 \
  --client-fraction 0.1 \
  --rounds 100 \
  --local-steps 4 \
  --learning-rate 0.01 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --scheduler-name cosine \
  --scheduler-step-size 10 \
  --scheduler-gamma 0.1 \
  --pretrained \
  --federated-checkpoint-path experiments/checkpoints/best_federated_non_iid_nc50_j4.pt \
  --last-federated-checkpoint-path experiments/checkpoints/last_federated_non_iid_nc50_j4.pt
```

The same pattern applies to the other combinations in the repository, whose artifact names include `j4`, `j8_r50`, and `j16_r25`.

### 4) main SparseSGDM experiments

The main sparse federated study uses IID data, `J=4`, 100 rounds, `sparsity-ratio=0.1`, `calibration-rounds=10`, and `calibration-batches=10`.

Supported mask strategies:

- `least_sensitive`
- `most_sensitive`
- `lowest_magnitude`
- `highest_magnitude`
- `random`

Example command for the least-sensitive strategy:

```bash
python main.py \
  --mode federated \
  --partition iid \
  --data-root ./data \
  --device auto \
  --seed 42 \
  --batch-size 32 \
  --val-fraction 0.1 \
  --num-workers 2 \
  --num-clients 100 \
  --client-fraction 0.1 \
  --rounds 100 \
  --local-steps 4 \
  --learning-rate 0.01 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --scheduler-name cosine \
  --scheduler-step-size 10 \
  --scheduler-gamma 0.1 \
  --pretrained \
  --use-sparse-sgd \
  --mask-strategy least_sensitive \
  --sparsity-ratio 0.1 \
  --calibration-rounds 10 \
  --calibration-batches 10 \
  --federated-checkpoint-path experiments/checkpoints/best_sparse_least_sensitive_sr0.1_cr10_cb10.pt \
  --last-federated-checkpoint-path experiments/checkpoints/last_sparse_least_sensitive_sr0.1_cr10_cb10.pt
```

The same structure applies to the other strategies:

```bash
# most_sensitive
python main.py ... --mask-strategy most_sensitive ...
# lowest_magnitude
python main.py ... --mask-strategy lowest_magnitude ...
# highest_magnitude
python main.py ... --mask-strategy highest_magnitude ...
# random
python main.py ... --mask-strategy random ...
```

The saved results in `experiments/checkpoints` include:

- `best_sparse_least_sensitive_sr0.1_cr10_cb10.pt`
- `best_sparse_most_sensitive_sr0.1_cr10_cb10.pt`
- `best_sparse_lowest_magnitude_sr0.1_cr10_cb10.pt`
- `best_sparse_highest_magnitude_sr0.1_cr10_cb10.pt`
- `best_sparse_random_sr0.1_cr10_cb10.pt`

### 5) sparsity-ratio and calibration-round studies

The repository contains additional sparse studies for:

- sparsity ratio `{0.05, 0.10, 0.20}`
- calibration rounds `{5,10}`

Example command for the sparsity study with the least-sensitive mask:

```bash
python main.py \
  --mode federated \
  --partition iid \
  --data-root ./data \
  --device auto \
  --seed 42 \
  --batch-size 32 \
  --val-fraction 0.1 \
  --num-workers 2 \
  --num-clients 100 \
  --client-fraction 0.1 \
  --rounds 100 \
  --local-steps 4 \
  --learning-rate 0.01 \
  --momentum 0.9 \
  --weight-decay 0.0005 \
  --scheduler-name cosine \
  --scheduler-step-size 10 \
  --scheduler-gamma 0.1 \
  --pretrained \
  --use-sparse-sgd \
  --mask-strategy least_sensitive \
  --sparsity-ratio 0.05 \
  --calibration-rounds 10 \
  --calibration-batches 10 \
  --federated-checkpoint-path experiments/checkpoints/best_sparse_least_sensitive_sr0.05_cr10_cb10.pt \
  --last-federated-checkpoint-path experiments/checkpoints/last_sparse_least_sensitive_sr0.05_cr10_cb10.pt
```

Similarly, replace `--sparsity-ratio` with `0.2` for the `0.2` study, and replace `--calibration-rounds` with `5` for the calibration-round study while keeping the rest of the configuration fixed.

The output files in `experiments/checkpoints` show the expected naming pattern, including:

- `best_sparse_least_sensitive_sr0.05_cr10_cb10.pt`
- `best_sparse_least_sensitive_sr0.2_cr10_cb10.pt`
- `best_sparse_least_sensitive_sr0.1_cr5_cb10.pt`

## Evaluation

Run the evaluation script to score every saved best checkpoint on the common CIFAR-100 test split:

```bash
python evaluate_checkpoints.py
```

This writes `experiments/checkpoint_metrics.csv` with fields:

- `checkpoint`
- `round_or_epoch`
- `validation_loss`
- `validation_accuracy`
- `test_loss`
- `test_accuracy`

## Plotting and output files

The repository includes plotting scripts that read existing experiment CSVs or checkpoint summaries and write PNGs to `experiments/`.

```bash
python export_federated_histories.py
python plot_centralized_curves.py
python plot_iid_vs_non_iid.py
python plot_results.py
python plot_mask_strategy.py
python plot_sparsity_ratio.py
python regenerate_sparse_figures.py
```

`regenerate_sparse_figures.py` is used to regenerate the final sparse comparison figures in the repository. 

Generated outputs include:

- `experiments/centralized_test_accuracy_curve.png`
- `experiments/centralized_test_loss_curve.png`
- `experiments/iid_vs_non_iid_test_accuracy.png`
- `experiments/validation_accuracy_comparison.png`
- `experiments/test_accuracy_comparison.png`
- `experiments/sparsity_ratio_comparison.png`
- `experiments/federated_histories/*.csv`
- `experiments/checkpoint_metrics.csv`

## Checkpoints and resume

The project saves:

- centralized best checkpoint: `experiments/checkpoints/best_centralized.pt`
- centralized latest checkpoint: `experiments/checkpoints/last_centralized.pt`
- federated best checkpoint: `experiments/checkpoints/best_federated_iid_j4.pt` or the corresponding non-IID / sparse variant
- federated latest checkpoint: `experiments/checkpoints/last_federated_iid_j4.pt` or the corresponding non-IID / sparse variant

Resume flags supported by the CLI are:

- `--resume` for centralized training
- `--federated-resume` for federated training

Example:

```bash
python main.py --mode centralized --epochs 200 --resume experiments/checkpoints/last_centralized.pt
python main.py --mode federated --partition iid --federated-resume experiments/checkpoints/last_federated_iid_j4.pt
```

These flags are validated in `main.py` and are for continuing interrupted runs, not for inventing a new training workflow.

## Reproducibility notes

The project seeds randomness in `main.py` with:

- Python `random.seed(seed)`
- NumPy `np.random.seed(seed)`
- PyTorch `torch.manual_seed(seed)`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

The dataset split is deterministic with the requested seed, and the federated client partitioning also uses the seed. The repository contains multiple-seed outputs for several settings, including:

- centralized seeds (`final_seed1_history.csv`, `final_seed2_history.csv`, `final_seed3_history.csv`)
- IID federated seeds (`best_federated_iid_j4_seed1.pt`, `..._seed2.pt`, `..._seed3.pt`)
- non-IID seeds (`*_seed43.pt`, `*_seed44.pt`)
- sparse runs (`*_seed43.pt`, `*_seed44.pt`)

This indicates that some experiments were repeated across multiple seeds, but the repository does not support a blanket statement that every result was repeated three times. The README only documents the configurations that are directly evidenced by the command-line interface and saved experiment artifacts.

## Project structure

```text
.
├── main.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── src/
├── evaluate_checkpoints.py
├── export_federated_histories.py
├── plot_centralized_curves.py
├── plot_iid_vs_non_iid.py
├── plot_mask_strategy.py
├── plot_results.py
├── plot_sparsity_ratio.py
├── regenerate_sparse_figures.py
├── experiments/
└── data/cifar-100-python/
```

## Important note

The repository contains both CLI defaults and report-specific experiment settings.

For reproduction of the paper results, follow the explicit commands above rather than the default main.py values alone.
