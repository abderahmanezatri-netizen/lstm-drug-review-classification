# Drug review classification with custom LSTM cells

Advanced Natural Language Processing project, 2026/2027.

We compare **Vanilla LSTM, CIFG and WMC** on English medication reviews from the UCI Drugs.com dataset. The cells are implemented with `nn.Linear`, following the professor's appendix, without `nn.LSTM`.

## Final experiment

The final experiment is in **`results/gpu_45min/`**. It uses 15,000 training reviews, 1,500 validation reviews and 3,000 test reviews. All 12 settings completed eight epochs. The recorded total runtime was **37 minutes 49 seconds** on an NVIDIA GeForce RTX 5060 Laptop GPU.

| Model | Parameters | Best epoch | Validation macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|---:|
| Vanilla | 1,185,219 | 7 | 0.5749 | 73.97% | 0.5727 |
| CIFG | 1,176,963 | 4 | 0.5715 | 75.63% | 0.5662 |
| WMC | 1,197,507 | 4 | 0.5810 | 74.60% | 0.5505 |
| Majority reference | 0 | - | - | 66.40% | 0.2660 |

WMC is the final model selected on validation macro F1. CIFG has the highest observed test accuracy; Vanilla has the highest test macro F1. Test scores were calculated after selection. Neutral reviews remain difficult for all three models.

## Setup and tests

Use Python 3.12 to match the recorded experiment, then install dependencies in your own environment:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

All five tests should pass. They cover the cell equations, gradients, parameter counts, deduplication, padding and the vectorized input transform.

On the original Windows laptop, the existing Conda environment is in the parent folder. From `projet_rnn`, use it directly; activation is optional:

```powershell
..\.venv\python.exe -m unittest discover -s tests -v
```

On another Windows laptop, create an environment inside the extracted project folder:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Predict a new review

The three selected checkpoints and vocabulary are included in the repository. The saved models can be tested without a GPU, the training dataset or another training run:

```sh
python -m src.predict --checkpoint results/gpu_45min/checkpoints/vanilla_class_weight.pt --text "This medicine helped me sleep but caused nausea."
python -m src.predict --checkpoint results/gpu_45min/checkpoints/cifg_class_weight.pt --text "This medicine helped me sleep but caused nausea."
python -m src.predict --checkpoint results/gpu_45min/checkpoints/wmc_class_weight.pt --text "This medicine helped me sleep but caused nausea."
```

On the original laptop, replace `python` with `..\.venv\python.exe`. On a friend's Windows laptop using the setup above, use `.\.venv\Scripts\python.exe`.

Use the same text to compare the models. The largest output probability gives the prediction. Mixed reviews can be ambiguous, and probabilities are not calibrated confidence estimates. The vocabulary is loaded from the checkpoint's experiment folder and checked against its saved hash.

## Dataset and labels

The original archive contains `drugsComTrain_raw.tsv` (161,297 reviews) and `drugsComTest_raw.tsv` (53,766 reviews). TSV means tab-separated values. The program reads it with Python's `csv.DictReader` and `delimiter='\t'`.

- [UCI dataset 462](https://archive.ics.uci.edu/dataset/462).
- [Archive used in this experiment](https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip).
- Local archive: `data/drugs.zip`.

The description page returned HTTP 403 when checked. The archive address in `src/data.py` is the download source used. The dataset link is also given in the professor's handout.

Ratings 1-4 are mapped to negative, 5-6 to neutral and 7-10 to positive. This mapping is a project decision. Only review text enters the classifier; ratings and other metadata are excluded from inputs.

HTML entities are decoded, text is lowercased and repeated whitespace is collapsed. Exact duplicate texts are removed, with training records taking precedence over identical test records. The first occurrence is retained even if ratings conflict. Near duplicates and patient/drug overlap are not ruled out.

Validation is taken from 15% of the cleaned official training source. The final training, validation and test subsets are stratified with seed 42 and shared by every run. The vocabulary is learned only from selected training reviews. Review hashes and the raw archive hash are recorded.

## Model and training choices

All cells use embeddings of size 64, hidden size 64, at most 160 tokens, masked mean pooling and dropout 0.2. Linear weights use Xavier uniform initialization and the forget bias starts at 1. Gradient norms are clipped at 1.0. Embeddings are learned from scratch.

Each cell is compared with four settings, all using Adam:

| Setting | Learning rate | Batch | Loss weights |
|---|---:|---:|---|
| Reference | 0.003 | 256 | None |
| Learning rate | 0.001 | 256 | None |
| Batch size | 0.003 | 128 | None |
| Weighted loss | 0.003 | 256 | `(N / (3 * n_class)) ** 0.5` |

Weights use training labels only. Each alternative changes one setting from the reference. Every model completes one epoch per round, and only complete rounds are retained. The best epoch and configuration are chosen using validation macro F1. The test is evaluated afterward.

This is a one-seed, reduced-data experiment. It does not establish statistical significance or measure hyperparameter interactions. Batch 128 gets more updates per epoch than batch 256, and equal hidden sizes do not give equal parameter counts.

## Reproduce training

Training is optional for testing the saved models. To repeat eight epochs with a time budget and a new output folder:

```sh
python -m src.train_fast --device cuda --minutes 45 --epochs 8 --output results/reproduce
```

Use `--device auto` to compare CPU and GPU first, or `--device cpu` without CUDA. Slower hardware can complete fewer rounds; inspect `budget.json`. The budget reserves time for evaluation but cannot guarantee a deadline if the system or network stalls.

UCI data is downloaded when `data/drugs.zip` is absent. Keep that archive for offline training. The program protects output directories containing `runs.json` against accidental overwriting.

## Files to inspect

| Path | Purpose |
|---|---|
| `src/models.py` | Manual cells and classifier |
| `src/data.py` | Data preparation and tokenization |
| `src/train_fast.py` | Final training experiment |
| `src/predict.py` | Prediction with saved weights |
| `tests/` | Five implementation checks |
| `results/gpu_45min/runs.json` | All 12 settings and epoch histories |
| `results/gpu_45min/results.json` | Full test results and selected checkpoints |
| `results/gpu_45min/comparison.csv` | Final model comparison |
| `results/gpu_45min/eda.json` | Counts, lengths and archive hash |
| `results/gpu_45min/environment.json` | Versions, hardware and settings |
| `results/gpu_45min/budget.json` | Duration and completed rounds |
| `output/gpu/hyperparameters.csv` | Validation comparison of 12 settings |
| `output/gpu/figures/` | Final charts and confusion matrices |
| `output/pdf/report.pdf` | Final five-page English report |
| `output/pdf/report.md` | Editable report text |
| `COMMANDES_DEMO_GPU.txt` | Original laptop's demo commands |

`src/train.py` supplies the evaluation helper used by the final trainer. Use `src/train_fast.py` to reproduce the final experiment. Colab notebooks and obsolete preliminary results have been removed.

## Build the report and sharing archive

These commands read saved results and do not retrain:

```sh
python tools/export_gpu_project.py
python tools/build_report.py
python tools/export_gpu_project.py
```

The first export generates the figures. The report builder reads the measured experiment. The last export includes the finished report in `output/gpu/Projet_RNN_GPU.zip`. This archive contains the code, tests, documentation, report, metrics, vocabulary and three selected checkpoints. It excludes the raw dataset and Python environments.

## Before submission

Fill in `team_members`, `lab_group` and `repository_url` in `report_details.json`, then run `python tools/build_report.py` to insert them in the PDF. The Markdown report is an editable text copy; the PDF builder reads the experiment and `report_details.json`, not changes made to the Markdown copy. Report body wording is in `tools/build_report.py`.

Create the public GitHub repository required by the professor. The PowerPoint is left for later. Do not upload `.venv/`, `.runtime/`, `data/` or `tmp/` to GitHub. The three selected demo checkpoints are included in Git (about 14 MB total). Other checkpoints, ZIP archives and preliminary runs are excluded. A clone can run the prediction commands after installing dependencies.

The handout's deadline is 25 October 2026 inclusive. The oral presentation is 10 minutes in English, at lab 6. The scripts do not publish a repository or submit the project.

## References

- Course handout: `Guidelines_RNN_2627.pdf`, pp. 1-2.
- UCI dataset 462 and archive linked above.
- Greff et al., [LSTM: A Search Space Odyssey](https://arxiv.org/abs/1503.04069), 2017. CIFG reading listed in the handout.
- [WMC reading listed in the handout](https://www.sciencedirect.com/science/article/pii/S0893608021003439). The implementation follows the supplied appendix.
