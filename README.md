# Hybrid Music Recommendation Using Collaborative and Content-Based Filtering

AIML339 final project

## Description
A two-tower neural reccommender whose item tower fuses a collaborative (CF) embedding with an audio-content embedding. 
It compares three fusion mechanisms (concatenation, weighted sum, gated), with and without CF-embedding dropout, against pure-CF and pre-content baselines. 
Evaluation uses a DropoutNet-style warm/ user-cold/ item-cold protocol over 30 seeds with Holm-corrected Wilcoxon tests.

## Features
- PyTorch two-tower retrieval model trained with BPR loss on implicit feedback
- Three fusion mechanisms: Concatenation, learned weighted sum, gated fusion
- CF-embedding dropout (rate 0.3) ablation, following DropoutNet
- Baselines: pure CF ('implicit' BPR matrix) matrix factorisation), pure content (audio k-NN), CF-only two-tower
- DropoutNet-style warm / user-cold / item-cold data splits, regenerated per seed (seeds 0-29)
- Metrics: Precision@10, Recall@10, NDCG@10, with paired Wilcoxon tests and Holm correction

## Dataset
**Music4All-Onion** (Moscati et al., CIKM 2022), CC BY 4.0:
https://zenodo.org/records/6609677

Download these three files from the Zenodo record:

| File | Size | Link |
|---|---|---|
| `id_mfcc_stats.tsv.bz2` | 93.8 MB | https://zenodo.org/records/6609677/files/id_mfcc_stats.tsv.bz2?download=1 |
| `userid_trackid_count.tsv.bz2` | 550.6 MB | https://zenodo.org/records/6609677/files/userid_trackid_count.tsv.bz2?download=1 |
| `id_genres_tf-idf.tsv.bz2` | 3.0 MB | https://zenodo.org/records/6609677/files/id_genres_tf-idf.tsv.bz2?download=1 |

The rest of the 15 GB record is not used.

## Installation
Requirements: a Google account, Google Colab (GPU runtime, T4 or L4), and about 1 GB of
free Google Drive space. The notebook reads the data from Google Drive and writes
results back to it, so it is intended to be run in Colab.

1. Clone or download this repository:

   git clone https://github.com/calizojoan/AIML339-Project/tree/main

2. Create a folder in Google Drive and upload the three data files above:

   MyDrive/hybrid_music_rec/
   ├── id_mfcc_stats.tsv.bz2
   ├── userid_trackid_count.tsv.bz2
   └── id_genres_tf-idf.tsv.bz2

3. Upload `AIML339project.ipynb` to Colab (or open it from Drive).
4. Dependencies: the notebook installs `implicit` (`pip install implicit`); PyTorch,
   NumPy, pandas, SciPy, statsmodels and scikit-learn are already available in Colab.

## Usage
1. Open `AIML339project.ipynb` in Colab 
2. Run the first cell to mount Google Drive and accept the permissions prompt:
```python
   from google.colab import drive
   drive.mount('/content/drive')
```
3. Check that the data path in the notebook points to your Drive folder from the
   Installation step.
4. Run the notebook top to bottom. Results are written to 'MyDrive/hybrid_music_rec/processed'.

## Results
Pre-computed results are in `results/`, so you can inspect them without re-running
the notebook.

| File | Contents |
|---|---|
| `cold_start_results.csv` | Per-seed, per-model, per-group Precision/Recall/NDCG@10 (30 seeds) |
| `report_table_precision_at_k.csv` | Mean ± std Precision@10 tables (Table I in the report) |
| `report_table_recall_at_k.csv` | Mean ± std Recall@10 |
| `report_table_ndcg_at_k.csv` | Mean ± std NDCG@10 |
| `significance_tests_precision_at_k.csv` | Wilcoxon tests with Holm correction on Precision@10 (used in the report) |
| `significance_tests.csv` | Same tests computed on NDCG@10 (robustness check) |
| `baseline_comparison.csv`, `baseline_pure_cf_results.csv` | Pure-CF baseline results |

## Repository Contents
```
AIML339project.ipynb   full pipeline: data prep, splits, models, evaluation, statistics
results/               result tables and per-seed outputs (see above)
README.md
.gitignore
```

## Settings
Embedding dimension 64; Adam, learning rate 0.005, weight decay 1e-6; batch size 1024;
500 steps per epoch; 60 epochs; BPR loss. `implicit` BPR baseline uses library defaults
(64 factors, learning rate 0.01, regularisation 0.01, 100 iterations). Data subsample:
the 2,000 most active users; 10% of users and 10% of items are cold in each seed.

## Limitations
Research prototype evaluated on synthetic cold-start splits of a 2,000-user subsample.
Epoch count was chosen from training-loss convergence rather than held-out validation.
Not a validated production recommender.

## Authors
Joanna Calizo, Victoria University of Wellington

## License
Academic project for AIML 339. The dataset (Music4All-Onion) is released under CC BY 4.0.
Libraries (`implicit`, PyTorch) are subject to their own licences.

