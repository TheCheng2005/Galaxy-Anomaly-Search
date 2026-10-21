"""Where the scripts look for inputs and write outputs.

Everything is anchored to the repo folder, so you can run scripts from anywhere.

  DATA_DIR      the image archive (.hdf5 cutouts + .parquet positions) built by dataset/pipeline.py.
                Default: a hubble_data/ folder next to this repo. Set HUBBLE_DATA_DIR to move it.
  SUBSET_DIR    where the scoring scripts write scores for the 20,000-image benchmark
                (TEST_MODE = True): generated/scores/. File names match paper_results/scores/.
  FULL_RUN_DIR  where they write scores for the whole archive (TEST_MODE = False).

Our own results live in paper_results/ and nothing writes there. Everything you make goes to
generated/ (git-ignored); set GENERATED_DIR to put it somewhere else.
"""
import os

ANALYSIS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('HUBBLE_DATA_DIR',
                          os.path.join(os.path.dirname(ANALYSIS_ROOT), 'hubble_data'))
STEM = '10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec'
HDF5_PATH = os.path.join(DATA_DIR, STEM + '.hdf5')
PARQUET_PATH = os.path.join(DATA_DIR, STEM + '.parquet')
INTERESTING_CSV_PATH = os.environ.get(
    'INTERESTING_CSV_PATH', os.path.join(ANALYSIS_ROOT, 'dataset', 'reference_anomalies.csv'))
GENERATED_DIR = os.environ.get('GENERATED_DIR', os.path.join(ANALYSIS_ROOT, 'generated'))
SUBSET_DIR = os.path.join(GENERATED_DIR, 'scores')
FULL_RUN_DIR = os.path.join(GENERATED_DIR, 'scores_full_archive')
