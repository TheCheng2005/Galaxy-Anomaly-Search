# Identifying Scientifically Interesting Galaxies with Vision-Language Models

Code for Cheng, Koblischke & Bovy (2026), Machine Learning and the Physical
Sciences Workshop, NeurIPS 2026.

We ask Gemini and Qwen to rate how interesting Hubble cutouts are, then check how many known
anomalies end up near the top of their rankings. This repo has everything to do that yourself,
plus the exact results behind the paper.

## What's where

```
dataset/        build the image archive (223,195 HST cutouts) + the list of known anomalies
scoring/        ask Gemini / Qwen to score the images (needs API keys, costs money)
catalogue/      pick the new candidate anomalies out of the scores
analysis/       Table 1, the Zoobot baseline, and the figures
common/         shared paths and a CSV reader
paper_results/  our results: scores, costs, the catalogue (see paper_results/README.md)
generated/      everything you produce (made on first run, git-ignored)
```

Your own runs never touch `paper_results/`, and ours never get in the way of yours. Each script
explains itself at the top: what it does, what you can change, what it costs and what to expect.

## Setup

Python 3.11+, run from this folder:

```bash
python3 -m pip install -r requirements.txt
cp .env.example .env    # only needed for dataset/pipeline.py
```

## Just want to check our numbers?

No API keys and no image archive needed. These read `paper_results/` and write to `generated/`:

```bash
python3 analysis/make_table1.py          # Table 1, Table 2 and the numbers in the text
python3 catalogue/build_catalogue.py     # the 102-candidate catalogue
python3 analysis/make_recall_figure.py   # Figure 1
```

## Running everything yourself

### 1. Get the images

```bash
python3 dataset/pipeline.py --work-dir work --output-dir ../hubble_data
```

This needs a free MAST CasJobs account (`CASJOBS_WSID`, `CASJOBS_PW` in `.env`) and takes a
while, since it downloads every HST frame once. If you already have the `.hdf5` and `.parquet`
files, put them in `../hubble_data/` or set `HUBBLE_DATA_DIR` and skip this step.

### 2. Score the images

```bash
export GEMINI_API_KEY=...       # for the gemini_* scripts
export OPENROUTER_API_KEY=...   # for the qwen_* scripts
python3 scoring/gemini_likert.py
python3 scoring/gemini_tournament.py
python3 scoring/qwen_likert.py
python3 scoring/qwen_tournament.py
```

By default each script runs in **test mode**, the paper's setup: it scores a 20,000-image
benchmark (all 167 known anomalies plus random background cutouts) instead of all 223,195
cutouts. Set `TEST_MODE = False` at the top of a script to score the whole archive, about 11×
the cost. In test mode, a seed-44 run costs roughly $10 (Gemini Likert), $7 (Gemini Tournament),
$13 (Qwen Likert) or $3.60 (Qwen Tournament). Gemini Likert runs seeds 44, 45 and 46 by default,
like the paper; set `RANDOM_SEEDS = [44]` to run just one.

Scores land in `generated/scores/`, named like the files in `paper_results/scores/`. The models
aren't fully deterministic, so expect recall close to the paper's (e.g. 98.0% ± 1.5% for Gemini
Likert across seeds), not image-for-image the same.

### 3. The Zoobot baseline

```bash
python3 analysis/zoobot_baseline.py
```

A conventional comparison (the Zoobot rows of Table 1): embeds the 20,000 images with a
pretrained galaxy model and ranks them by two unsupervised anomaly scores. CPU only, about ten
minutes, needs the `.hdf5` archive.

### 4. Tables, catalogue and figures

```bash
python3 analysis/make_table1.py
python3 catalogue/build_catalogue.py
python3 analysis/make_recall_figure.py
python3 analysis/plot_appendix_catalog.py   # needs the .hdf5 archive
```

These read `paper_results/` by default. To run them on your own scores, point them at
`generated/scores/` (each script says how at the top).

The exact montage in the manuscript (Figure 2) was drawn with
`matplotlib==3.6.3 numpy==1.26.4 Pillow==10.2.0 pandas==2.1.4 h5py==3.16.0`.

## Citation

See `CITATION.cff`.

## License

MIT; see `LICENSE`.
