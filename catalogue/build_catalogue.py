"""Build the released catalogue of 102 candidate anomalies (Section 3, Appendix B).

Starting from the Gemini Likert scores (seed 44), it keeps images that score 45 or more and
aren't one of the 167 reference anomalies (138 images), then removes:
  - matches within 3 arcsec of the interacting-galaxy catalogue of O'Ryan et al. (2023)
    (138 -> 105),
  - matches within 3 arcsec of a galaxy mentioned in astronomy papers, from the galaxy-mentions
    dataset, huggingface.co/datasets/astronolan/galaxy-mentions (105 -> 104),
  - two objects we found by hand to be already studied, ESO 255-7 and SDSS J1205+4910 (-> 102).

  python3 catalogue/build_catalogue.py

By default it uses the catalogue matches we saved (paper_results/catalogue/catalogue_matches.csv),
so it runs without the external catalogues and gives exactly paper_results/catalogue/candidates.csv.
To redo the matching yourself, pass both --oryan-csv and --galaxy-mentions-parquet.

To build a catalogue from your own run, point --scores at it, e.g.
  python3 catalogue/build_catalogue.py --scores generated/scores/gemini_likert_seed44.csv \
      --oryan-csv ... --galaxy-mentions-parquet ...
You'll need the external catalogues then, because the saved matches only cover our 138 images.

Output: generated/candidates.csv, one row per candidate, highest score first.
"""
import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import scores_csv

SCORES = ROOT / 'paper_results/scores/gemini_likert_seed44.csv'
SUBSET = ROOT / 'paper_results/benchmark_subset.csv'
SELECTION = ROOT / 'paper_results/catalogue/catalogue_matches.csv'
FIELDS = ['filename', 'imagescore', 'interesting', 'SourceRA', 'SourceDec',
          'oryan_match', 'galaxy_mentions_match']


def positional_matches(records, catalogue, ra, dec):
    import numpy as np
    from scipy.spatial import cKDTree

    def vectors(r, d):
        r, d = np.deg2rad(np.asarray(r, float)), np.deg2rad(np.asarray(d, float))
        return np.column_stack((np.cos(d)*np.cos(r), np.cos(d)*np.sin(r), np.sin(d)))

    points = vectors([r['SourceRA'] for r in records], [r['SourceDec'] for r in records])
    coordinates = catalogue.dropna(subset=[ra, dec])
    tree = cKDTree(vectors(coordinates[ra], coordinates[dec]))
    radius = 2*np.sin(np.deg2rad(3/3600)/2)
    return [bool(hits) for hits in tree.query_ball_point(points, radius)]


def select_candidates(scores_path=SCORES, subset_path=SUBSET, selection_path=SELECTION,
                      oryan_csv=None, galaxy_mentions_parquet=None):
    with Path(subset_path).open() as f:
        positions = {r['filename']: r for r in csv.DictReader(f)}
    rows = []
    for r in scores_csv.read(scores_path):
        score = int(float(r['imagescore']))
        if score >= 45 and str(r['interesting']) == '0':
            p = positions[r['filename']]
            rows.append(dict(filename=r['filename'], imagescore=score, interesting=0,
                             SourceRA=p['SourceRA'], SourceDec=p['SourceDec']))
    rows.sort(key=lambda r: (-r['imagescore'], r['filename']))
    if oryan_csv:
        import pandas as pd
        oryan = pd.read_csv(oryan_csv)
        mentions = pd.read_parquet(galaxy_mentions_parquet)
        mentions = mentions.loc[mentions.has_resolved_coordinates.eq(True)]
        first = positional_matches(rows, oryan, 'RA', 'Dec')
        second = positional_matches(rows, mentions, 'resolved_ra_deg', 'resolved_dec_deg')
    else:
        with Path(selection_path).open() as f:
            saved = list(csv.DictReader(f))
        flags = {r['filename']: r for r in saved}
        first = [flags[r['filename']]['oryan_match'] == '1' for r in rows]
        second = [flags[r['filename']]['galaxy_mentions_match'] == '1' for r in rows]
    for r, a, b in zip(rows, first, second):
        r.update(oryan_match=a, galaxy_mentions_match=b)
    after_oryan = [r for r in rows if not r['oryan_match']]
    retained = [r for r in after_oryan if not r['galaxy_mentions_match']]
    counts = [len(rows), len(after_oryan), len(retained)]
    retained = [r for r in retained if r['filename'] not in {'4000904587828', '4001415925883'}]  # ESO 255-7 and SDSS J1205+4910
    return retained, counts + [len(retained)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, default=SCORES)
    parser.add_argument('--subset', type=Path, default=SUBSET)
    parser.add_argument('--selection', type=Path, default=SELECTION)
    parser.add_argument('--oryan-csv', type=Path)
    parser.add_argument('--galaxy-mentions-parquet', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'generated/candidates.csv')
    args = parser.parse_args()
    rows, counts = select_candidates(args.scores, args.subset, args.selection,
                                     args.oryan_csv, args.galaxy_mentions_parquet)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print('Catalogue selection: ' + ' -> '.join(map(str, counts)))
    print(f'Wrote {len(rows)} candidates to {args.output}')


if __name__ == '__main__':
    main()
