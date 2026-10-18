# Identifying Scientifically Interesting Galaxies with Vision-Language Models

Code for Cheng, Koblischke & Bovy (2026), Machine Learning and the Physical
Sciences Workshop, NeurIPS 2026.

Run from this folder with Python 3.11+. Steps 3–5 use the bundled paper results;
new model runs are stochastic. Outputs go to `generated/`.

```bash
python3 -m pip install -r requirements.txt
```

## 1. Build the dataset

Query HSC/MAST, deduplicate, and make cutouts. Set `CASJOBS_WSID` and
`CASJOBS_PW` in the environment or `.env` (template: `.env.example`).

```bash
python3 dataset/pipeline.py --work-dir work --output-dir ../hubble_data
```

For existing data, set `HUBBLE_DATA_DIR` to the HDF5/Parquet directory.

## 2. Score the images

Set `GEMINI_API_KEY` and `OPENROUTER_API_KEY`. Prompts, seeds, and settings are
in each script; new scores go to `generated/results/subset_test/`.

```bash
python3 scoring/gemini_likert.py
python3 scoring/gemini_tournament.py
python3 scoring/qwen_likert.py
python3 scoring/qwen_tournament.py
```

## 3. Select candidates

Apply the score cutoff, saved catalogue matches, and two manual exclusions.

```bash
python3 catalogue/build_catalogue.py
```

## 4. Calculate tables

Read saved scores and usage; calculate performance, costs, and seed variability.

```bash
python3 analysis/make_table1.py
python3 analysis/zoobot_baseline.py
```

`zoobot_baseline.py` computes the Zoobot rows of Table 1 and needs the parent HDF5 archive.

## 5. Draw figures

Plot recall and the candidate montage. The montage needs the parent HDF5 archive.

```bash
python3 analysis/make_recall_figure.py
python3 analysis/plot_appendix_catalog.py
```

Exact manuscript montage rendering uses a separate environment with
`matplotlib==3.6.3 numpy==1.26.4 Pillow==10.2.0 pandas==2.1.4 h5py==3.16.0`.

## Citation

See `CITATION.cff`.

## License

MIT; see `LICENSE`.
