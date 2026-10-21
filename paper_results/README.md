# Paper results

These are the exact files behind the numbers, tables and figures in the paper. They're kept here
so you can check our results and redraw the figures without spending anything on API calls.

Nothing in the repo writes to this folder. When you run the code yourself, your files go to
`generated/` instead, with the same file names, so you can compare the two side by side.

## The benchmark

**`benchmark_subset.csv`** lists the 20,000 cutouts we scored: all 167 that match a reference
anomaly, plus 19,833 random background cutouts. It's the set you get by running a scoring script
in test mode with seed 44 (seeds 45 and 46 draw a different background).

| column | meaning |
|---|---|
| `index` | position in the run (the order the images were shuffled into) |
| `hdf5_index` | row of the cutout in the image archive (`.hdf5`) |
| `filename` | the cutout's HSC SourceID |
| `interesting` | 1 if it matches a reference anomaly (within 3 arcsec), else 0 |
| `classification` | the reference anomaly's class (merger, lens, ...); empty for background |
| `SourceRA`, `SourceDec` | position in degrees |

## Scores (`scores/`)

One file per run. Each row is one cutout, sorted from highest to lowest score. The VLM files all
have `filename`, `imagescore` and `interesting` (as above); some also have the
`classification` and position columns. `qwen_tournament_seed44.csv` is the raw output, with
capitalised headers and a token-usage footer (lines starting with `#`). The analysis code reads
both formats.

| file | what it is | in the paper |
|---|---|---|
| `gemini_likert_seed44.csv` | Gemini 3.1 Flash-Lite, Likert, score 0–50 | Table 1, Figure 1, the catalogue |
| `gemini_likert_seed45.csv`, `gemini_likert_seed46.csv` | the same with two more seeds (different background draw) | the seed spread (Section 3) |
| `gemini_likert_high_thinking_seed44.csv` | Gemini Likert with thinking set to high instead of low | high vs. low thinking (Section 2) |
| `gemini_tournament_seed44.csv` | Gemini, Tournament, score = rounds survived, 0–10 | Table 1, Figure 1 |
| `qwen_likert_seed44.csv` | Qwen3.5-397B-A17B, Likert, 0–50 | Table 1, Figure 1 |
| `qwen_tournament_seed44.csv` | Qwen, Tournament, 0–10 | Table 1, Figure 1 |
| `zoobot_nano_scores.csv` | the conventional baselines on frozen Zoobot nano embeddings: `isolation_forest` (seed 0) and `knn_distance` (k = 10); higher = more unusual | Table 1 |

## Costs

**`token_usage.csv`** holds the tokens each VLM run used: `input`, `output`, `thinking` (Gemini
only). For Qwen, `cost_usd` is what OpenRouter charged. Gemini costs are computed from the
tokens at $0.25 / $1.50 per million input / output tokens, with thinking billed as output.
`run` matches the file names in `scores/`. Used for the cost columns of Tables 1 and 2.

## The candidate catalogue (`catalogue/`)

**`candidates.csv`** is the released catalogue: the 102 cutouts with a Gemini Likert score of
45 or more that aren't reference anomalies, aren't in either literature catalogue, and weren't
removed by hand (Section 3, Appendix B). `catalogue/build_catalogue.py` rebuilds it exactly.

| column | meaning |
|---|---|
| `filename` | the cutout's HSC SourceID |
| `imagescore` | Gemini Likert score, 45–50 |
| `interesting` | always 0 (none are reference anomalies) |
| `SourceRA`, `SourceDec` | position in degrees |
| `oryan_match`, `galaxy_mentions_match` | always False here (matches were removed) |

**`catalogue_matches.csv`** records, for all 138 images that scored 45 or more, whether each one
matched the interacting-galaxy catalogue of O'Ryan et al. (2023) (`oryan_match`, 33 images) or
the galaxy-mentions dataset (`galaxy_mentions_match`, 2 images, one of them also in O'Ryan)
within 3 arcsec (1 = match). It lets `build_catalogue.py` run without those external catalogues.

## Figure 1 examples (`figure1_examples/`)

The three images in Figure 1(c), named by SourceID, already cropped and zoomed for the figure.
`analysis/make_recall_figure.py --fits-dir DIR` remakes them from the original HST frames.

## Not in here

The list of reference anomalies (the 1,176 objects from O'Ryan et al. 2025 that define
`interesting`) is an input rather than a result, so it lives at
`dataset/reference_anomalies.csv`.
