"""Conventional baseline: rank the 20,000-image benchmark by unsupervised anomaly scores in
Zoobot's embedding space (no training on our data), reported in the format of Table 1.

Embeddings: the Zoobot ConvNeXt-nano greyscale encoder (Walmsley et al.), pretrained on Galaxy
Zoo volunteer answers. Cutouts are upsampled from 150 to 224 px, pixels scaled to 0-1, and the
encoder returns 640 pooled features per image.

Anomaly scores (higher = more unusual):
  isolation forest  how quickly random axis-aligned splits isolate an image (Liu et al. 2008)
  kNN distance      mean cosine distance to the image's k nearest neighbours in the benchmark

    python analysis/zoobot_baseline.py --hdf5 PATH/10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec.hdf5
"""
import argparse
import csv
import io
import sys
import time
from pathlib import Path

import numpy as np

ANALYSIS = Path(__file__).resolve().parent
ROOT = ANALYSIS.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ANALYSIS))
from common.paths import HDF5_PATH  # noqa: E402
from make_table1 import REVIEW_SET_SIZE, recall_curve  # noqa: E402

SUBSET = ROOT / 'results/paper/gemini_low_seed44/subset.csv'
ENCODER = 'hf_hub:mwalmsley/zoobot-encoder-greyscale-convnext_nano'
SIDE = 224


def load_cutouts(hdf5_path, indices):
    """Decode the benchmark cutouts (greyscale uint8, 150 x 150) in subset order."""
    import h5py
    from PIL import Image
    order = np.argsort(indices)  # read the archive in file order, then restore subset order
    images = np.empty((len(indices), 150, 150), np.uint8)
    with h5py.File(hdf5_path, 'r') as archive:
        blobs = archive['images']
        for k in order:
            with Image.open(io.BytesIO(bytes(blobs[indices[k]]))) as image:
                images[k] = np.asarray(image.convert('L'))
    return images


def embed(images, batch=128):
    """Zoobot nano features (N x 640) on the CPU."""
    import timm
    import torch
    import torch.nn.functional as F
    model = timm.create_model(ENCODER, pretrained=True, in_chans=1, num_classes=0).eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(images), batch):
            x = torch.from_numpy(images[start:start + batch]).float().div_(255).unsqueeze(1)
            out.append(model(F.interpolate(x, size=SIDE, mode='bilinear')).numpy())
    return np.concatenate(out)


def isolation_forest_scores(features, seed):
    from sklearn.ensemble import IsolationForest
    forest = IsolationForest(n_estimators=1000, max_samples=256, random_state=seed).fit(features)
    return -forest.score_samples(features)  # score_samples is high for normal points


def knn_scores(features, k):
    from sklearn.neighbors import NearestNeighbors
    distances, _ = NearestNeighbors(n_neighbors=k + 1, metric='cosine').fit(features).kneighbors(features)
    return distances[:, 1:].mean(axis=1)  # column 0 is the image itself


def evaluate(names, scores, truth):
    """Table 1 columns for the top-REVIEW_SET_SIZE images (continuous scores, so no ties)."""
    curve = recall_curve(dict(zip(names, scores)), dict(zip(names, truth)), int(sum(truth)))
    top = np.argsort(-scores)[:REVIEW_SET_SIZE]
    found = int(np.asarray(truth)[top].sum())
    return dict(n_selected=REVIEW_SET_SIZE, recall=found / sum(truth), precision=found / REVIEW_SET_SIZE,
                expected_recall=curve[REVIEW_SET_SIZE - 1], expected_recall_at_300=curve[299])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--hdf5', type=Path, default=Path(HDF5_PATH))
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'generated' / 'zoobot_baseline')
    parser.add_argument('--k', type=int, default=10, help='neighbours for the kNN distance')
    parser.add_argument('--seeds', type=int, default=5, help='isolation-forest seeds (spread reported)')
    parser.add_argument('--threads', type=int, default=8, help='CPU threads for the encoder')
    args = parser.parse_args()
    import torch
    torch.set_num_threads(args.threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with SUBSET.open() as f:
        rows = list(csv.DictReader(f))
    names = [r['filename'] for r in rows]
    truth = [int(r['interesting']) for r in rows]
    cache = args.output_dir / 'embeddings_nano.npy'
    t0 = time.time()
    if cache.exists():
        features = np.load(cache)
        print(f'Loaded cached embeddings {features.shape}')
    else:
        images = load_cutouts(args.hdf5, np.array([int(r['hdf5_index']) for r in rows]))
        print(f'Decoded {len(images):,} cutouts in {time.time() - t0:.0f} s')
        features = embed(images)
        np.save(cache, features)
        print(f'Embedded {features.shape} in {time.time() - t0:.0f} s')

    t1 = time.time()
    forests = [isolation_forest_scores(features, seed) for seed in range(args.seeds)]
    knn = knn_scores(features, args.k)
    print(f'Scored in {time.time() - t1:.0f} s')

    results = {'Isolation forest': evaluate(names, forests[0], truth), f'kNN distance (k={args.k})': evaluate(names, knn, truth)}
    spread = [evaluate(names, s, truth)['expected_recall'] for s in forests]
    print(f"\nZoobot nano, {len(names):,} images, {sum(truth)} reference anomalies, top {REVIEW_SET_SIZE:,}")
    print(f"  {'Method':26}{'N Sel':>7}{'Recall':>8}{'Prec':>7}{'ExpR@%d' % REVIEW_SET_SIZE:>10}{'ExpR@300':>10}")
    for method, m in results.items():
        print(f"  {method:26}{m['n_selected']:>7}{m['recall']:>8.3f}{m['precision']:>7.3f}"
              f"{m['expected_recall']:>10.3f}{m['expected_recall_at_300']:>10.3f}")
    print(f"  Isolation forest over {args.seeds} seeds: Expected Recall@{REVIEW_SET_SIZE} "
          f"{np.mean(spread):.3f} +/- {np.std(spread, ddof=1):.3f}")

    with (args.output_dir / 'scores.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'interesting', 'isolation_forest', 'knn_distance'])
        writer.writerows(zip(names, truth, np.round(forests[0], 6), np.round(knn, 6)))
    tex = []
    for method, m in results.items():
        tex.append(f"    Zoobot nano & \\textit{{{method}}} & Top {REVIEW_SET_SIZE:,} & "
                   f"{REVIEW_SET_SIZE:,} & {m['recall']:.3f} & {m['precision']:.3f} & "
                   f"{m['expected_recall']:.3f} & \\$0 \\\\".replace(',', '{,}'))
    (args.output_dir / 'table1_rows.tex').write_text('\n'.join(tex) + '\n')
    print(f'\nWrote scores.csv and table1_rows.tex to {args.output_dir}')


if __name__ == '__main__':
    main()
