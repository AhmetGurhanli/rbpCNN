This reposotory is a supporting information for the academic paper titled "rbpCNN: A physics-informed deep learning model for predicting piRNA and mRNA interactions"

## Citation

If you use this code, please cite the associated rbpCNN aricle given below:

Gürhanlı, A., Nematzadeh, S., Çevik, T. et al. rbpCNN: a biophysics-informed deep learning model for predicting piRNA and mRNA interactions. Sci Rep (2026). https://doi.org/10.1038/s41598-026-48797-5

## DATASETS 

The datasets are taken from http://cosbi2.ee.ncku.edu.tw/data_download/piRNA_mRNA_binding These datasets are online resources of the paper "Yang, TH., Shiue, SC., Chen, KY. et al. Identifying piRNA targets on mRNAs in C. elegans using a deep multi-head attention network. BMC Bioinformatics 22, 503 (2021). https://doi.org/10.1186/s12859-021-04428-6". Some columns are omitted in the version used in this work, to make the data file smaller.

## Content
This repository contains the code for training and evaluating **rbpCNN**, a lightweight biophysics-informed convolutional neural network for piRNA–mRNA interaction prediction.

The model uses a 21-channel interaction tensor containing:
- 16 pair-identity channels,
- 1 base-pair compatibility channel,
- 2 helix-run channels,
- 1 positional-difference channel,
- 1 structural accessibility channel.

## Repository files

```text
.
├── rbpCNN_21channel.py          # Train rbpCNN with 5-fold cross-validation on WT CLASH
├── eval_independent.py          # Evaluate a trained model on independent CSR-1 CLASH data
├── WT_CLASH_positive.csv        # Required training positive set
├── WT_CLASH_negative.csv        # Required training negative set
├── CSR-1_CLASH_positive.csv     # Required independent positive set
└── CSR-1_CLASH_negative.csv     # Required independent negative set
```

The four CSV dataset files must be placed in the same folder as the Python scripts.

## Required dataset file names

The training script expects:

```text
WT_CLASH_positive.csv
WT_CLASH_negative.csv
```

The independent evaluation script expects:

```text
CSR-1_CLASH_positive.csv
CSR-1_CLASH_negative.csv
```

Each CSV file should contain one piRNA sequence column and one target-site / mRNA sequence column.

Accepted piRNA column names:

```text
piRNA_seq
pirna_seq
piRNA_sequence
pirna
```

Accepted mRNA / target-site column names:

```text
site_seq
mRNA_seq
mrna_seq
target_seq
target_site
mRNA_sequence
```

The scripts automatically detect the first matching column name.

## Python environment

Python 3.9 or newer is recommended.

Install the required packages:

```bash
pip install numpy pandas scikit-learn matplotlib torch
```

Optional but useful:

```bash
pip install psutil
```

If you want GPU acceleration, install the PyTorch version compatible with your CUDA version from the official PyTorch installation page.

## 1. Train rbpCNN on WT CLASH

Place these files in the repository folder:

```text
rbpCNN_21channel.py
WT_CLASH_positive.csv
WT_CLASH_negative.csv
```

Run:

```bash
python rbpCNN_21channel.py
```

The script performs 5-fold cross-validation using the WT CLASH positive and negative sets.

Default training settings in the script include:
- 5-fold stratified cross-validation,
- batch size 64,
- maximum 30 epochs,
- early stopping patience 4,
- learning rate 1e-3,
- dropout 0.50,
- weight decay 1e-4,
- helix-run kernel size 7.

## Training outputs

After training, the script saves:

```text
best_fold1.pt
best_fold2.pt
best_fold3.pt
best_fold4.pt
best_fold5.pt
cv_fold_metrics_struct_delta21.csv
oof_predictions_struct_delta21.csv
learning_curves.tiff
roc_curves.tiff
learning_curve_epoch1_train_only.tiff
nussinov_pos.pkl
nussinov_neg.pkl
```

Important files:

- `best_fold1.pt` to `best_fold5.pt`: trained model weights for each fold.
- `cv_fold_metrics_struct_delta21.csv`: per-fold cross-validation metrics.
- `oof_predictions_struct_delta21.csv`: out-of-fold predictions.
- `nussinov_pos.pkl` and `nussinov_neg.pkl`: cached structural accessibility predictions.

The Nussinov cache files are generated automatically. If they already exist, the script reuses them to avoid recomputing structural features.

## 2. Evaluate on independent CSR-1 CLASH

After training, make sure the following files are in the same folder:

```text
eval_independent.py
best_fold1.pt
CSR-1_CLASH_positive.csv
CSR-1_CLASH_negative.csv
```

Run:

```bash
python eval_independent.py
```

By default, the evaluation script:
- loads `best_fold1.pt`,
- loads the CSR-1 positive and negative files,
- evaluates with threshold 0.50,
- saves prediction results and an ROC curve.

## Independent evaluation outputs

The evaluation script saves:

```text
independent_predictions.csv
independent_roc.png
```

It also prints the following metrics to the console:

```text
AUC
Accuracy
Precision
Recall
F1-score
Threshold used
```

## Optional evaluation arguments

You can change the batch size:

```bash
python eval_independent.py --batch_size 64
```

You can change the decision threshold:

```bash
python eval_independent.py --thr 0.45
```

You can compute a Youden-optimal threshold on the independent set:

```bash
python eval_independent.py --youden
```

Note: using `--youden` selects an optimal threshold on the test set itself. This can be useful for exploratory analysis, but for a strict independent evaluation, the default fixed threshold should be used.

## Typical full workflow

```bash
# 1. Train on WT CLASH
python rbpCNN_21channel.py

# 2. Evaluate fold-1 model on CSR-1 CLASH
python eval_independent.py
```

## Reproducibility

The scripts set the random seed to:

```text
42
```

for Python, NumPy, and PyTorch.

GPU use is automatic if CUDA is available. Otherwise, the scripts run on CPU.

## Notes

- The training script uses fixed input lengths inferred from the cleaned dataset sequences.
- The expected benchmark setting uses 21-nt piRNA sequences and 31-nt target-site windows.
- Structural accessibility is estimated using a Nussinov-style paired/unpaired mask and cached for later reuse.
- The independent evaluation script uses the same 21-channel tensor construction as the training script.


