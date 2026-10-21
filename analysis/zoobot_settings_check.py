"""Check that the Zoobot baselines don't depend on their settings (Section 2, "Conventional Baselines").

The paper uses common defaults: 10 neighbours for the kNN distance, and an isolation forest of
1,000 trees, each grown on 256 images. This script reruns both with other settings:

  kNN distance      10, 50 and 100 neighbours
  isolation forest  1,000 or 2,000 trees; 64, 256 or 1,024 images per tree (5 seeds each)

  python3 analysis/zoobot_settings_check.py

It reuses the embeddings saved by analysis/zoobot_baseline.py, so run that first. No GPU, no
image archive needed after that; it takes a few minutes on a CPU.

What you get
  generated/zoobot_baseline/settings_check.csv, one row per setting: recall and precision in the
  top 2,000 and Expected Recall@300 (means over seeds for the forest, with the spread of the
  recall). Ours is paper_results/zoobot_settings_check.csv.

What to expect
  Recall in the top 2,000 stays within 2 percentage points of the defaults: kNN 85.0% with 10
  neighbours, 85.6% with 50 or 100; isolation forest 69.5-71.9% across settings (71.4% with the
  defaults, mean of 5 seeds). In the top 300 the kNN distance drops a little with more
  neighbours (18.0% with 10, 15.0% with 100).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))
from zoobot_baseline import ROOT, SUBSET, evaluate, isolation_forest_scores, knn_scores  # noqa: E402

NEIGHBOURS = [10, 50, 100]
FORESTS = [(1000, 256), (2000, 256), (1000, 64), (1000, 1024)]  # (trees, images per tree); first is the default
SEEDS = 5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'generated' / 'zoobot_baseline')
    args = parser.parse_args()
    cache = args.output_dir / 'embeddings_nano.npy'
    if not cache.exists():
        sys.exit(f'No embeddings at {cache}: run analysis/zoobot_baseline.py first.')
    features = np.load(cache)
    with SUBSET.open() as f:
        rows = list(csv.DictReader(f))
    names = [r['filename'] for r in rows]
    truth = [int(r['interesting']) for r in rows]

    table = []
    for k in NEIGHBOURS:
        m = evaluate(names, knn_scores(features, k), truth)
        table.append(dict(method='kNN distance', setting=f'k = {k}', seeds=1, recall=m['recall'], recall_sd='',
                          precision=m['precision'], expected_recall_at_300=m['expected_recall_at_300']))
    for trees, samples in FORESTS:
        runs = [evaluate(names, isolation_forest_scores(features, seed, trees, samples), truth) for seed in range(SEEDS)]
        mean = lambda key: float(np.mean([r[key] for r in runs]))  # noqa: E731
        table.append(dict(method='Isolation forest', setting=f'{trees} trees, {samples} images per tree', seeds=SEEDS,
                          recall=mean('recall'), recall_sd=float(np.std([r['recall'] for r in runs], ddof=1)),
                          precision=mean('precision'), expected_recall_at_300=mean('expected_recall_at_300')))

    print(f"  {'Method':18}{'Setting':36}{'Recall':>8}{'+/-':>7}{'Prec':>7}{'ExpR@300':>10}")
    for r in table:
        sd = f"{r['recall_sd']:.3f}" if r['recall_sd'] != '' else ''
        print(f"  {r['method']:18}{r['setting']:36}{r['recall']:>8.3f}{sd:>7}{r['precision']:>7.3f}"
              f"{r['expected_recall_at_300']:>10.3f}")
    output = args.output_dir / 'settings_check.csv'
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows({key: round(v, 4) if isinstance(v, float) else v for key, v in r.items()} for r in table)
    print(f'\nWrote {output}')


if __name__ == '__main__':
    main()
