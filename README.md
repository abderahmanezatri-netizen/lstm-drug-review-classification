# Drug Review Sentiment Classification

Comparison of Vanilla LSTM, Coupled Input-Forget Gate (CIFG), and Working Memory Connections (WMC) for three-class sentiment classification with PyTorch.

The recurrent cells implement their equations using `nn.Linear`, without the built-in `nn.LSTM`.

## Dataset

[UCI Drug Review Dataset (Drugs.com), dataset 462](https://archive.ics.uci.edu/dataset/462).
The [original archive](https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip) contains the official training and test TSV files. The loader downloads it into `data/drugs.zip` when needed.

Review ratings define the targets: 1-4 negative, 5-6 neutral, and 7-10 positive. Only review text is used as model input. HTML entities and whitespace are normalized, text is lowercased, and exact duplicates are removed across the official splits, with training records taking precedence.

Validation is drawn from the cleaned official training set. The recorded experiment uses shared stratified subsets of 15,000 training, 1,500 validation and 3,000 test reviews, with seed 42. The vocabulary is built from training text only.

## Installation

Python 3.12 was used for the recorded experiment.

```sh
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```sh
# Linux / macOS
source .venv/bin/activate
```

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

GPU training requires a CUDA-compatible PyTorch installation. Prediction works on CPU.

## Models and experimental protocol

All architectures use 64-dimensional embeddings, 64 hidden units, a maximum sequence length of 160 tokens, masked mean pooling and dropout 0.2. Linear weights use Xavier initialization; forget biases start at 1. Gradients are clipped to norm 1.

Each architecture is evaluated with four configurations:

| Configuration | Adam learning rate | Batch size | Class weights |
|---|---:|---:|---|
| Reference | 0.003 | 256 | None |
| Learning rate | 0.001 | 256 | None |
| Batch size | 0.003 | 128 | None |
| Weighted loss | 0.003 | 256 | Square root of inverse class frequency |

All 12 configurations completed eight epochs. Checkpoints and configurations were selected using validation macro F1; test evaluation followed selection. The trainer retains complete epoch rounds across all configurations under a configurable runtime budget.

## Results

| Architecture | Trainable parameters | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|
| Vanilla LSTM | 1,185,219 | 0.5749 | 73.97% | 0.5727 |
| CIFG | 1,176,963 | 0.5715 | 75.63% | 0.5662 |
| WMC | 1,197,507 | 0.5810 | 74.60% | 0.5505 |
| Majority baseline | 0 | - | 66.40% | 0.2660 |

WMC was selected on validation macro F1. CIFG obtained the highest test accuracy, while Vanilla obtained the highest test macro F1. These are observations from one seed, rather than evidence of a universally superior architecture. Neutral reviews remain difficult; they represent about 9% of the selected data.

Metrics, histories, vocabulary, split hashes and environment information are in `results/gpu_45min/`. Comparison plots and confusion matrices are in `output/gpu/figures/`; `output/gpu/hyperparameters.csv` records the configuration comparison.

## Prediction

The three selected checkpoints are included. Prediction does not require downloading the dataset or retraining.

```sh
python -m src.predict --checkpoint results/gpu_45min/checkpoints/vanilla_class_weight.pt --text "This medicine helped me sleep but caused nausea."
python -m src.predict --checkpoint results/gpu_45min/checkpoints/cifg_class_weight.pt --text "This medicine helped me sleep but caused nausea."
python -m src.predict --checkpoint results/gpu_45min/checkpoints/wmc_class_weight.pt --text "This medicine helped me sleep but caused nausea."
```

Outputs are probabilities for negative, neutral and positive sentiment. The vocabulary is loaded from the checkpoint experiment folder.

## Reproduce training

```sh
python -m src.train_fast --device cuda --minutes 45 --epochs 8 --output results/reproduce
```

Use `--device cpu` without CUDA, or `--device auto` to benchmark available devices. The runtime budget can result in fewer complete epochs on slower hardware. Use a new output directory to preserve existing results.

## Source structure

- `src/data.py`: dataset loading, preprocessing, vocabulary and splits.
- `src/models.py`: manually implemented recurrent cells and classifier.
- `src/train_fast.py`: architecture and configuration comparison.
- `src/train.py`: evaluation helper and alternative training entry point.
- `src/predict.py`: inference from a checkpoint.
- `src/baseline.py`: non-recurrent reference implementations.
- `tests/`: cell equations, gradients, parameter counts, padding and deduplication checks.

## Limitations

This experiment uses one random seed, a reduced training subset and truncated sequences. Labels are derived from ratings and may not reflect every nuance of a review. Exact deduplication does not eliminate near duplicates. Additional seeds and larger-data experiments would be needed to assess the stability of the architecture rankings.

## References

- UCI Drug Review Dataset (Drugs.com), linked above.
- Greff et al., [LSTM: A Search Space Odyssey](https://arxiv.org/abs/1503.04069).
- [Working Memory Connections reading](https://www.sciencedirect.com/science/article/pii/S0893608021003439).
