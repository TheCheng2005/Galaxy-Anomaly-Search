"""Portable input and scoring-output paths, anchored to the release folder."""
import os

ANALYSIS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get('HUBBLE_DATA_DIR',
                          os.path.join(os.path.dirname(ANALYSIS_ROOT), 'hubble_data'))
STEM = '10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec'
HDF5_PATH = os.path.join(DATA_DIR, STEM + '.hdf5')
PARQUET_PATH = os.path.join(DATA_DIR, STEM + '.parquet')
INTERESTING_CSV_PATH = os.environ.get(
    'INTERESTING_CSV_PATH', os.path.join(ANALYSIS_ROOT, 'dataset', 'reference_anomalies.csv'))
RESULTS_ROOT = os.environ.get('PAPER_RESULTS_DIR', os.path.join(ANALYSIS_ROOT, 'generated'))
SUBSET_DIR = os.path.join(RESULTS_ROOT, 'results', 'subset_test')
FULL_RUN_DIR = os.path.join(RESULTS_ROOT, 'results', 'full_catalog')
