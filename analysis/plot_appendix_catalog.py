"""Draw Figure 2 (Appendix B): the 30 highest-scoring of the 102 catalogue candidates.

  python3 analysis/plot_appendix_catalog.py

It reads paper_results/catalogue/candidates.csv and takes the cutouts from the image archive
(the .hdf5 file, see dataset/pipeline.py), so you need that file. Images are ordered by score,
then by ID. It writes generated/appendix_catalog.pdf and .png, and the LaTeX caption.
To plot your own catalogue, point CATALOGUE (below) at the folder with your candidates.csv.
"""
import argparse
import io
import sys
import h5py
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common.paths import HDF5_PATH

CATALOGUE = ROOT / 'paper_results' / 'catalogue'
N_PANELS = 30


def main(output_dir, hdf5_path):
    output_dir.mkdir(parents=True, exist_ok=True)
    catalogue = pd.read_csv(CATALOGUE/'candidates.csv', dtype={'filename':str})
    sample = catalogue.sort_values(['imagescore','filename'],ascending=[False,True]).head(N_PANELS).copy()
    subset = pd.read_csv(ROOT/'paper_results/benchmark_subset.csv', dtype={'filename':str}).set_index('filename')
    images = {}
    with h5py.File(hdf5_path, 'r') as archive:
        for filename in sample.filename:
            index = int(subset.loc[filename, 'hdf5_index'])
            with Image.open(io.BytesIO(bytes(archive['images'][index]))) as image:
                images[filename] = image.convert('RGB')
    fig,axes = plt.subplots(5,6,figsize=(8.4,7.5))
    for ax,row in zip(axes.flat,sample.itertuples()):
        image = images[row.filename]
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'{row.filename}\n{row.imagescore}/50',fontsize=7,pad=3,linespacing=1.15)
        for spine in ax.spines.values():
            spine.set_edgecolor('#dddddd')
            spine.set_linewidth(.35)
    fig.subplots_adjust(left=.018,right=.982,bottom=.012,top=.966,wspace=.11,hspace=.35)
    fig.savefig(output_dir/'appendix_catalog.pdf',bbox_inches='tight',pad_inches=.03)
    fig.savefig(output_dir/'appendix_catalog.png',dpi=200,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)
    caption = (rf'\caption{{The {len(sample)} highest-scoring images among the {len(catalogue)} retained candidates with Gemini '
               r'\textit{Likert} scores $\geq45$ outside AnomalyMatch, after excluding matches to existing literature '
               r'catalogs within $3\arcsec$ and two sources identified as previously studied through manual literature review.}')
    (output_dir/'appendix_caption.tex').write_text(caption+'\n')
    print(f'Saved {len(sample)} panels; scores {sample.imagescore.max()} to {sample.imagescore.min()}, to {output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'generated')
    parser.add_argument('--hdf5', type=Path, default=Path(HDF5_PATH), help='External parent image archive')
    args = parser.parse_args()
    main(args.output_dir, args.hdf5)
