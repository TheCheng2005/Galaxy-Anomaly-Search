"""Draw Figure 1: (a) recall against the number of images selected, (b) the Gemini Likert score
histogram, and (c) three example images.

  python3 analysis/make_recall_figure.py

It reads the four seed-44 runs and the Zoobot nano kNN-distance ranking (zoobot_nano_scores.csv) in
paper_results/scores/ and writes generated/recall_figure.png and .svg. The example panels (c) come from paper_results/figure1_examples/. To redraw them from the
original HST frames instead, pass --fits-dir DIR (missing frames are downloaded from MAST).

To plot your own runs, point SCORES (below) at generated/scores/.
"""
import argparse
import csv
import sys
import io
import shutil
from urllib.request import urlopen
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from matplotlib.ticker import PercentFormatter, StrMethodFormatter

# Model, method, colour, marker.
METHODS = [
    ('qwen', 'likert', '#146c2e', 'D'),
    ('gemini', 'likert', '#0b3d91', 's'),
    ('qwen', 'tournament', '#8fd9a8', '^'),
    ('gemini', 'tournament', '#7fb8f5', 'o'),
]
NAMES = {'qwen': 'Qwen-3.5-397B', 'gemini': 'Gemini 3.1 Flash-Lite'}


ANALYSIS = Path(__file__).resolve().parent
ROOT = ANALYSIS.parent
sys.path.insert(0, str(ROOT))
from common import scores_csv
SCORES = ROOT / 'paper_results' / 'scores'
RESULTS = {(model, method): SCORES / f'{model}_{method}_seed44.csv' for model, method, *_ in METHODS}
ZOOBOT = SCORES / 'zoobot_nano_scores.csv'   # from analysis/zoobot_baseline.py
MAX_SELECTED = 2500
ZOOBOT_COLOR = '#cc5e0a'


def load_counts(path):
    """Read one run into score histograms for each reference label."""
    rows = scores_csv.read(path)
    scores = np.array([float(r['imagescore']) for r in rows])
    labels = np.array([int(r['interesting']) for r in rows])
    counts = np.array([np.bincount(scores[labels == k].astype(int), minlength=51) for k in (0, 1)])
    return counts


def zoobot_recall(path):
    """Recall in the top N images of the Zoobot kNN-distance ranking, for N = 1, 2, ... (no ties)."""
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    scores = np.array([float(r['knn_distance']) for r in rows])
    labels = np.array([int(r['interesting']) for r in rows])
    return np.cumsum(labels[np.argsort(-scores, kind='stable')]) / labels.sum()


def rendered_crop(image, low, high):
    """Reproduce the original 800px Matplotlib crop before the figure's zoom/pan."""
    fig = plt.figure(figsize=(5, 5), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(image, interpolation='bilinear')
    ax.set(xlim=(low, high), ylim=(high, low))
    ax.set_axis_off()
    fig.canvas.draw()
    result = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()).copy()).convert('RGB')
    plt.close(fig)
    return result


# SourceID, HAP product, RA/Dec, cutout size. First two follow the canonical pipeline.
SUBJECTS = [
    ('4001415925883', 'hst_10494_15_acs_wfc_f814w_j9c715_drc.fits', 181.41836151856464, 49.17477080083705, 150),
    ('4000838861232', 'hst_10429_19_acs_wfc_f814w_j95t19_drc.fits', 202.0090294649435, -31.515268355607187, 150),
    ('4000743731958', 'hst_10405_10_acs_wfc_f814w_j96v10_drc.fits', 334.403219, .835206, 600),
]


def example_images(folder):
    """Download full FITS once, extract the subjects, and apply the display framing."""
    from astropy.io import fits
    from astropy.nddata import Cutout2D
    from astropy.visualization import ZScaleInterval, simple_norm
    from astropy.wcs import WCS

    folder.mkdir(parents=True, exist_ok=True)
    images = []
    for ident, name, ra, dec, size in SUBJECTS:
        path = folder / name
        if not path.exists():
            print(f'Downloading {name}', flush=True)
            url = f'https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/{name}'
            with urlopen(url, timeout=180) as response, path.open('wb') as stream:
                shutil.copyfileobj(response, stream)
        with fits.open(path, memmap=True) as hdul:
            hdu = hdul['SCI']
            center = WCS(hdu.header).world_to_pixel_values(ra, dec)
            data = Cutout2D(hdu.data, center, size, mode='strict', copy=True).data
        if size == 150:
            # Canonical ZScale and JPEG quality, then the figure's JPEG conversion.
            lo, hi = ZScaleInterval(contrast=.05).get_limits(data)
            pixels = np.rint(np.clip((data.astype(float)-lo)/(hi-lo), 0, 1)*255).astype('uint8')
            image = Image.fromarray(pixels)
            for quality in (95, 90):
                buffer = io.BytesIO()
                image.save(buffer, 'JPEG', quality=quality)
                image = Image.open(io.BytesIO(buffer.getvalue())).convert('RGB')
        else:
            scaled = simple_norm(data, 'asinh', percent=99.5)(data)
            pixels = (np.nan_to_num(np.clip(scaled, 0, 1))*255).astype('uint8')
            image = Image.fromarray(pixels[::-1]).convert('RGB')
        images.append(image)
    images[0] = rendered_crop(images[0], 30, 120)
    images[2] = rendered_crop(images[2].crop((215, 171, 373, 329)), 4, 154)
    for image, (zoom, pan_x, pan_y, flip) in zip(images, [(1.25, 0, 15, False), (1.2, 0, -97, False), (1.15, 93, 76, True)]):
        w, h = image.size
        side = min(w, h) / zoom
        left, top = (w-side)/2 * (1+pan_x/100), (h-side)/2 * (1+pan_y/100)
        crop = image.transform((800, 800), Image.Transform.EXTENT,
                               (left, top, left+side, top+side), Image.Resampling.BICUBIC)
        yield crop.transpose(Image.Transpose.FLIP_TOP_BOTTOM) if flip else crop


def cached_example_images():
    for ident, *_ in SUBJECTS:
        with Image.open(ROOT / 'paper_results' / 'figure1_examples' / f'{ident}.png') as image:
            yield image.copy()


def main(output, fits_dir=None, zoobot_style='scatter', zoobot_every=5):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 24,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#333333', 'axes.linewidth': 1.2,
                         'axes.labelsize': 27, 'axes.axisbelow': True})
    # Pixel geometry of the 4239×2530 canvas; both plot heights = 924px.
    fig = plt.figure(figsize=(4239/200, 2530/200), dpi=200, facecolor='white')
    def axes(x, y, w, h):
        return fig.add_axes([x/4239, (2530-y-h)/2530, w/4239, h/2530])
    a, b = axes(468, 168, 2893, 924), axes(468, 1368, 2893, 924)
    for model, method, color, marker in METHODS:
        counts = load_counts(RESULTS[model, method])
        if model == 'gemini' and method == 'likert':
            histogram = counts
        cumulative = counts[:, ::-1].cumsum(axis=1)[:, ::-1]
        selected, recall = cumulative.sum(axis=0), cumulative[1] / counts[1].sum()
        cutoffs = [10] if method == 'tournament' else [t for t in range(50, -1, -1) if selected[t] <= MAX_SELECTED]
        dark = '#146c2e' if model == 'qwen' else '#0b3d91'
        a.scatter(selected[cutoffs], recall[cutoffs], marker=marker, color=color,
                  s=240 if method == 'tournament' else 70, linewidth=.8, zorder=4,
                  edgecolor=dark if method == 'tournament' else 'white',
                  label=f'{NAMES[model]} {method.title()}')
        for t in ([10] if method == 'tournament' else [33 if model == 'gemini' else 36]):
            text = 'Round 10' if method == 'tournament' else f'Score≥{t}'
            # Positions in plot coordinates keep labels in open space below the data.
            label_positions = {
                ('gemini', 'likert', 33): (2290, .77),
                ('qwen', 'likert', 36): (1780, .76),
                ('gemini', 'tournament', 10): (1900, .66),
                ('qwen', 'tournament', 10): (2200, .60),
            }
            a.annotate(text, (selected[t], recall[t]),
                       xytext=label_positions[model, method, t], textcoords='data',
                       ha='center', va='top', fontsize=21, color=dark,
                       arrowprops=dict(arrowstyle='-', color='#666666', lw=.7,
                                       shrinkA=5,
                                       shrinkB=9 if method == 'tournament' else 5))
    zoobot = zoobot_recall(ZOOBOT)[:MAX_SELECTED]
    label = 'Zoobot nano kNN distance'
    if zoobot_style == 'line':
        a.plot(np.arange(1, len(zoobot) + 1), zoobot, color=ZOOBOT_COLOR, linewidth=2.6, zorder=3, label=label)
    else:   # one point per anomaly recovered (where the curve steps up), every zoobot_every-th of them
        steps = np.flatnonzero(np.diff(np.r_[0, zoobot]) > 0)[zoobot_every - 1::zoobot_every]
        a.scatter(steps + 1, zoobot[steps], marker='o', color=ZOOBOT_COLOR, s=79, linewidth=.8,
                  edgecolor='white', zorder=3, label=label)
    a.set(xlim=(0, MAX_SELECTED), ylim=(-.025, 1.04), xticks=np.arange(0, MAX_SELECTED + 1, 500), yticks=np.arange(0, 1.01, .2))
    a.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    a.xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    a.set_xlabel('Top N images in the ranking', labelpad=10)
    a.set_ylabel('Fraction of anomalies\nrecovered')
    a.set_title(f'Fraction of {histogram[1].sum():g} anomalies recovered from {histogram.sum()/1000:g}k images', fontsize=29, pad=34)
    a.grid(color='#dddddd')
    a.axhline(1, color='#888888', linewidth=1.4, zorder=2, clip_on=False)
    a.legend(loc='lower right', fontsize=22, framealpha=1, borderpad=.35, labelspacing=.25, handletextpad=.4, handlelength=1)

    for k, color, label in [(0, '#94a3b8', 'Background'), (1, '#146c2e', 'Anomaly')]:
        b.bar(np.arange(51) + (k-.5)*.38, histogram[k], .38, color=color,
              edgecolor='white', linewidth=.8, label=label, zorder=3)
    b.set(yscale='log', ylim=(.6, 75000), xlim=(-1, 51), xticks=range(0, 51, 5))
    b.set_xlabel('Image Score (0–50) — Gemini 3.1 Flash-Lite', labelpad=10)
    b.set_ylabel('Image Count', labelpad=12)

    # Derive cutoff metrics from the same seed-44 histogram shown in panel (b).
    cutoff = 33
    selected_count = histogram[:, cutoff:].sum()
    recovered = histogram[1, cutoff:].sum()
    recall = recovered / histogram[1].sum()
    precision = recovered / selected_count
    cutoff_edge = cutoff - .5  # Include both bars at score 33.
    b.axvline(cutoff_edge, ymax=.80, color='black', linestyle='--', linewidth=1.8, zorder=4)
    b.text(cutoff_edge + .4, .80,
           f'Score≥{cutoff}\nRecall: {recall:.1%}\nPrecision: {precision:.1%}',
           transform=b.get_xaxis_transform(), ha='left', va='top',
           fontsize=22, color='black', linespacing=1.3,
           bbox=dict(facecolor='white', edgecolor='none', alpha=.9, pad=2), zorder=5)
    b.grid(axis='y', which='major', color='#dddddd', linewidth=1)
    b.grid(axis='y', which='minor', color='#eeeeee', linewidth=.6)
    b.legend(loc='upper center', fontsize=22, ncol=2, framealpha=1, edgecolor='#cccccc', borderpad=.35, handletextpad=.4)

    images = example_images(fits_dir) if fits_dir is not None else cached_example_images()
    for i, image in enumerate(images):
        ax = axes(3458.73, 92.814 + i*749.76, 721.54, 721.185)
        ax.imshow(image, interpolation='bilinear')
        ax.set(xticks=[], yticks=[])
        for spine in ax.spines.values():
            spine.set(visible=True, color='#555555', linewidth=.7*2640/1700)
        ax.text(.045, .95, ['i)', 'ii)', 'iii)'][i], transform=ax.transAxes,
                color='white', fontsize=21*2640/1700, va='top')
    for label, x, y in [('(a)', 65, 68), ('(b)', 65, 1436), ('(c)', 3450, 68)]:
        fig.text(x/4239, (2530-y)/2530, label, fontsize=72*72/200)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('.png', '.svg'):
        fig.savefig(str(output) + ext, dpi=200)
    plt.close(fig)
    print(f'Saved {output}.png and {output}.svg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=Path, default=ROOT / 'generated' / 'recall_figure')
    parser.add_argument('--fits-dir', type=Path, help='Regenerate example panels from FITS in this folder (downloads missing products)')
    parser.add_argument('--zoobot-style', choices=['line', 'scatter'], default='scatter',
                        help='Zoobot kNN in panel (a): a point per anomaly recovered (the paper), or its recall curve')
    parser.add_argument('--zoobot-every', type=int, default=5, help='with scatter: keep every K-th point (paper: 5)')
    args = parser.parse_args()
    main(args.output, args.fits_dir, args.zoobot_style, args.zoobot_every)
