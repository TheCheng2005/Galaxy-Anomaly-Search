#!/usr/bin/env python3
"""Build the image archive the paper scores: 223,195 HST cutouts (Section 2, "Dataset construction").

You only need this if you want to rebuild the archive from scratch. If you already have the
.hdf5 and .parquet files, put them in ../hubble_data/ (or set HUBBLE_DATA_DIR) and skip it.

What it does, step by step
  1. Asks MAST for every HST ACS/WFC F814W image processed by the Hubble Advanced Products
     pipeline (HAP-SVM drizzled frames).
  2. Queries the Hubble Source Catalog v3 on MAST CasJobs for extended-source detections on those
     images, up to 10 million rows (LIMIT). This is the slow, server-side part.
  3. Removes near-duplicates: keeps one detection per 10 arcsec (RADIUS_ARCSEC), preferring
     unsaturated detections without quality flags. That leaves 223,195 detections, saved with
     their positions as the .parquet file.
  4. Downloads each FITS frame, cuts a 150 x 150 px cutout (7.5 arcsec) around every kept
     detection, stretches it with ZScale (contrast 0.05, as in AnomalyMatch), and stores it as a
     JPEG in the .hdf5 file, named by its HSC SourceID. Each frame is deleted after cutting.

  python3 dataset/pipeline.py --work-dir work --output-dir ../hubble_data

What you need
  A free MAST CasJobs account: put CASJOBS_WSID (your numeric WSID) and CASJOBS_PW in .env
  (copy .env.example). Plenty of time and bandwidth, since step 4 downloads every frame once.

Handy options
  --stop-after catalog   stop after step 2 (the raw detection table, in --work-dir)
  --stop-after parquet   stop after step 3 (positions only, no images)
  If a later step fails, rerunning starts again from step 1.

What you get
  ../hubble_data/10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec.parquet  (positions and
  catalogue columns) and .hdf5 (the cutouts, in the same order). MAST sometimes reprocesses
  frames, so a rebuild today can differ slightly from the paper's archive.
"""

import argparse
import csv
import gc
import io
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astroquery.mast import Mast, Observations
from dotenv import load_dotenv
from mastcasjobs import MastCasJobs
from PIL import Image
from scipy.spatial import cKDTree


CASJOBS = "https://mastweb.stsci.edu/ps1casjobs/services/jobs.asmx"
PRODUCT_RE = re.compile(
    r"^mast:HST/product/(?P<filename>(?P<image_name>hst_[^_]+_[^_]+_acs_wfc_f814w)_"
    r"[a-z0-9]{6}_drc\.fits)$",
    re.IGNORECASE,
)

LIMIT = 10_000_000
RADIUS_ARCSEC = 10.0
SIZE = 150
JPEG_QUALITY = 95
ZCONTRAST = 0.05
STEM = "10m_dedup_hsc_acs_wfc_f814w_0000_minsep10p0arcsec"

RAW_COLUMNS = [
    "MatchID", "SourceID", "SourceRA", "SourceDec", "XImage", "YImage",
    "ImageName", "Instrument", "Detector", "Filter", "CI", "KronRadius",
    "Flags", "Det",
]
FLOAT_COLUMNS = {"SourceRA", "SourceDec", "XImage", "YImage", "CI", "KronRadius"}
RAW_TRANSPORT_COLUMNS = [
    f"CONVERT(varchar(32), [{name}], 3) AS [{name}]"
    if name in FLOAT_COLUMNS else f"[{name}]"
    for name in RAW_COLUMNS
]
RAW_SCHEMA = pa.schema([
    ("MatchID", pa.int64()), ("SourceID", pa.int64()),
    ("SourceRA", pa.float64()), ("SourceDec", pa.float64()),
    ("XImage", pa.float64()), ("YImage", pa.float64()),
    ("ImageName", pa.large_string()), ("Instrument", pa.large_string()),
    ("Detector", pa.large_string()), ("Filter", pa.large_string()),
    ("CI", pa.float64()), ("KronRadius", pa.float64()),
    ("Flags", pa.int64()), ("Det", pa.large_string()),
])
FINAL_SCHEMA = pa.schema(list(RAW_SCHEMA) + [
    pa.field("fits_path", pa.large_string()),
    pa.field("original_shard_index", pa.int64()),
    pa.field("is_saturated", pa.bool_()),
    pa.field("shard_index", pa.int32()),
])


def mast_inventory(work: Path) -> pd.DataFrame:
    products = Mast.service_request(
        "Mast.Caom.Filtered",
        {
            "columns": "dataURL",
            "filters": [
                {"paramName": "obs_collection", "values": ["HST"]},
                {"paramName": "provenance_name", "values": ["HAP-SVM"]},
                {"paramName": "instrument_name", "values": ["ACS/WFC"]},
                {"paramName": "filters", "values": ["F814W"]},
            ],
        },
        pagesize=50_000,
    )

    fits_dir = (work / "fits").resolve()
    found: list[dict[str, object]] = []
    ignored = 0
    for value in products["dataURL"]:
        uri = str(value)
        match = PRODUCT_RE.fullmatch(uri)
        if not match:
            ignored += 1
            continue
        filename = match.group("filename")
        found.append({
            "image_name": match.group("image_name").lower(),
            "path": str(fits_dir / filename),
        })
    inventory = pd.DataFrame(found).drop_duplicates().sort_values(
        "image_name", kind="mergesort"
    ).reset_index(drop=True)
    print(f"Wrote MAST inventory: {len(inventory):,} products ({ignored:,} rows ignored by regex)")
    return inventory


def catalog_sql(image_table: str, output_table: str, limit: int, lo: int, hi: int) -> str:
    return f"""
SELECT TOP {int(limit)}
    d.MatchID,
    d.SourceID,
    d.SourceRA,
    d.SourceDec,
    d.Ximage AS XImage,
    d.Yimage AS YImage,
    d.ImageName,
    d.Instrument,
    d.Detector,
    d.Filter,
    d.CI,
    d.KronRadius,
    d.Flags,
    d.Det
INTO MyDB.{output_table}
FROM dbo.DetailedCatalog AS d
JOIN MyDB.{image_table} AS i
    ON d.ImageName = CAST(i.ImageName AS varchar(128))
WHERE
    i.Seq BETWEEN {int(lo)} AND {int(hi)}
    AND d.Det = 'Y'
    AND d.Detector = 'ACS/WFC'
    AND d.Filter = 'F814W'
    AND (d.Flags % 2) = 1
""".strip()


def typed_csv_table(text: str, schema: pa.Schema) -> pa.Table:
    first = text.splitlines()[0]
    names = []
    for item in next(csv.reader([first])):
        match = re.match(r"\[(.+)]\s*:[^:]+$", item)
        names.append(match.group(1) if match else item.strip().strip("[]"))
    table = pacsv.read_csv(
        pa.py_buffer(text.encode()),
        read_options=pacsv.ReadOptions(column_names=names, skip_rows=1),
        convert_options=pacsv.ConvertOptions(
            column_types={field.name: field.type for field in schema},
            null_values=["NULL", "null"],
            strings_can_be_null=True,
        ),
    )
    table = table.select(schema.names).cast(schema)
    columns = []
    for field, column in zip(schema, table.columns, strict=True):
        if pa.types.is_large_string(field.type):
            column = pc.utf8_trim_whitespace(column)
            if field.name == "ImageName":
                column = pc.utf8_lower(column)
        columns.append(column)
    return pa.Table.from_arrays(columns, schema=schema)


def run_job(cj: MastCasJobs, query: str, context: str, task: str, estimate: int = 120) -> None:
    job_id = cj.submit(query, context=context, task_name=task, estimate=estimate)
    print(f"Submitted CasJobs job {job_id}: {task}")
    cj.monitor(job_id, timeout=15)


def query_counts(
    cj: MastCasJobs,
    inventory: pd.DataFrame,
    image_table: str,
    output: str,
) -> pd.DataFrame:
    query = f"""
SELECT i.Seq, CAST(i.ImageName AS varchar(128)) AS ImageName,
       COUNT_BIG(d.SourceID) AS [RowCount]
INTO MyDB.{output}
FROM MyDB.{image_table} AS i
LEFT JOIN dbo.DetailedCatalog AS d
    ON d.ImageName = CAST(i.ImageName AS varchar(128))
    AND d.Det = 'Y'
    AND d.Detector = 'ACS/WFC'
    AND d.Filter = 'F814W'
    AND (d.Flags % 2) = 1
GROUP BY i.Seq, CAST(i.ImageName AS varchar(128))
""".strip()
    run_job(cj, query, "HSCv3", "gethst_counts")
    count_schema = pa.schema([
        ("Seq", pa.int64()), ("ImageName", pa.large_string()), ("RowCount", pa.int64())
    ])
    parts = []
    for lo in range(0, len(inventory), 1000):
        text = cj.quick(
            f"SELECT Seq, ImageName, [RowCount] FROM [{output}] "
            f"WHERE Seq BETWEEN {lo} AND {lo + 999} ORDER BY Seq",
            astropy=False, context="MYDB",
        )
        parts.append(typed_csv_table(text, count_schema))
    table = pa.concat_tables(parts)
    counts = table.to_pandas()
    cj.quick(f"DROP TABLE IF EXISTS [{output}]", astropy=False, context="MYDB")
    print(f"Wrote HSC counts: {int(counts['RowCount'].sum()):,} eligible rows")
    return counts


def make_batch_plan(counts: pd.DataFrame, target_rows: int) -> list[dict[str, object]]:
    # Preserve five-image query boundaries at the ten-million-row limit.
    groups = []
    remaining = LIMIT
    for lo in range(0, len(counts), 5):
        hi = min(lo + 5, len(counts))
        rows = int(counts.iloc[lo:hi]["RowCount"].sum())
        boundary = rows > remaining
        group = {
            "lo": lo, "hi": hi - 1, "rows": min(rows, remaining),
            "query_limit": remaining,
        }
        groups.append((group, boundary))
        remaining -= min(rows, remaining)
        if boundary or remaining == 0:
            break
    batches: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    for group, boundary in groups:
        if boundary:
            if current is not None:
                batches.append(current)
                current = None
            batches.append(group.copy())
            break
        if current is not None and int(current["rows"]) + int(group["rows"]) > target_rows:
            batches.append(current)
            current = None
        if current is None:
            current = group.copy()
        else:
            current["hi"] = group["hi"]
            current["rows"] = int(current["rows"]) + int(group["rows"])
    if current is not None:
        batches.append(current)
    for i, batch in enumerate(batches):
        batch["index"] = i
    return batches


def retrieve_batch(
    cj: MastCasJobs,
    batch: dict[str, object],
    image_table: str,
    table_prefix: str,
    page_rows: int,
    writer: pq.ParquetWriter,
) -> None:
    index = int(batch["index"])
    expected = int(batch["rows"])
    output = f"{table_prefix}_o{index:03d}"
    query = catalog_sql(
        image_table, output, int(batch["query_limit"]), int(batch["lo"]), int(batch["hi"])
    )
    run_job(cj, query, "HSCv3", f"gethst_catalog_{index:03d}")
    index_sql = f"CREATE UNIQUE CLUSTERED INDEX IX_{output} ON [{output}] (SourceID)"
    run_job(cj, index_sql, "MYDB", f"gethst_index_{index:03d}", 30)

    downloaded = 0
    last_source = -1
    while downloaded < expected:
        tables = []
        segment_rows = 0
        while downloaded + segment_rows < expected and segment_rows < 100_000:
            want = min(page_rows, expected - downloaded - segment_rows)
            text = cj.quick(
                f"SELECT TOP {want} {', '.join(RAW_TRANSPORT_COLUMNS)} FROM [{output}] "
                f"WHERE SourceID > {last_source} ORDER BY SourceID",
                astropy=False, context="MYDB",
            )
            page = typed_csv_table(text, RAW_SCHEMA)
            last_source = int(page["SourceID"][-1].as_py())
            tables.append(page)
            segment_rows += len(page)
        segment = pa.concat_tables(tables)
        writer.write_table(segment)
        downloaded += len(segment)
        print(f"Catalog chunk {index}: {downloaded:,}/{expected:,}")
    cj.quick(f"DROP TABLE IF EXISTS [{output}]", astropy=False, context="MYDB")
    print(f"Wrote catalog chunk {index}: {expected:,} rows")


def build_raw_catalog(
    cj: MastCasJobs,
    counts: pd.DataFrame,
    work: Path,
    image_table: str,
    table_prefix: str,
    target_rows: int,
    page_rows: int,
) -> Path:
    raw = work / "hsc_acswfc_f814w_extended_raw.parquet"
    plan = make_batch_plan(counts, target_rows)
    print(f"Catalog plan: {len(plan)} batches")
    with pq.ParquetWriter(raw, RAW_SCHEMA, compression="snappy", version="2.6") as writer:
        for batch in plan:
            retrieve_batch(cj, batch, image_table, table_prefix, page_rows, writer)
    cj.quick(f"DROP TABLE IF EXISTS [{image_table}]", astropy=False, context="MYDB")
    print(f"Wrote {raw}: {sum(batch['rows'] for batch in plan):,} rows")
    return raw


def sky_vectors(ra: np.ndarray, dec: np.ndarray) -> np.ndarray:
    ra_r = np.deg2rad(ra.astype(float, copy=False))
    dec_r = np.deg2rad(dec.astype(float, copy=False))
    return np.column_stack((
        np.cos(dec_r) * np.cos(ra_r),
        np.cos(dec_r) * np.sin(ra_r),
        np.sin(dec_r),
    ))


def deduplicate(raw: Path, inventory: pd.DataFrame, output: Path) -> pd.DataFrame:
    print("Reading the four deduplication columns")
    priority = pq.read_table(raw, columns=["SourceID", "SourceRA", "SourceDec", "Flags"])
    source_id = priority["SourceID"].to_numpy()
    ra = priority["SourceRA"].to_numpy()
    dec = priority["SourceDec"].to_numpy()
    flags = priority["Flags"].to_numpy()
    del priority
    print("Building the 10M-row sky cKDTree")
    xyz = sky_vectors(ra, dec)
    tree = cKDTree(xyz)
    saturated = (flags & 0b110) != 0
    order = np.lexsort((source_id, flags, saturated.astype(np.int8)))
    blocked = np.zeros(len(source_id), dtype=bool)
    keep: list[int] = []
    radius = 2 * np.sin(np.deg2rad(RADIUS_ARCSEC / 3600) / 2)
    print("Applying the greedy 10-arcsec suppression")
    for processed, idx in enumerate(order, 1):
        if not blocked[idx]:
            keep.append(int(idx))
            blocked[tree.query_ball_point(xyz[idx], r=radius)] = True
        if processed % 500_000 == 0:
            print(f"Dedup: processed={processed:,}, kept={len(keep):,}")
    keep_sorted = np.sort(np.asarray(keep, dtype=np.int64))
    del tree, xyz, order, blocked, source_id, ra, dec, flags, saturated
    gc.collect()

    path_by_image = dict(zip(inventory["image_name"], inventory["path"], strict=True))
    parts = []
    offset = 0
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=250_000):
        end = offset + batch.num_rows
        lo = int(np.searchsorted(keep_sorted, offset, side="left"))
        hi = int(np.searchsorted(keep_sorted, end, side="left"))
        if hi > lo:
            selected = pa.Table.from_batches([batch]).take(pa.array(keep_sorted[lo:hi] - offset))
            frame = selected.to_pandas()
            frame["fits_path"] = frame["ImageName"].map(path_by_image)
            frame["original_shard_index"] = keep_sorted[lo:hi]
            frame["is_saturated"] = (frame["Flags"] & 0b110) != 0
            parts.append(frame)
        offset = end
    result = pd.concat(parts, ignore_index=True)
    result = result.sort_values(["ImageName", "SourceID"], kind="mergesort").reset_index(drop=True)
    result["shard_index"] = np.arange(len(result), dtype=np.int32)
    table = pa.Table.from_pandas(result[FINAL_SCHEMA.names], schema=FINAL_SCHEMA, preserve_index=False)
    pq.write_table(
        table, output, compression="snappy", version="2.6",
        row_group_size=max(len(table), 1), use_dictionary=True,
    )
    print(f"Wrote {output}: {len(result):,} rows")
    return result


def first_2d_hdu(hdul: fits.HDUList) -> int:
    return next(
        index for index, hdu in enumerate(hdul)
        if getattr(getattr(hdu, "data", None), "ndim", None) == 2
    )


def stretch_to_uint8(array: np.ndarray) -> np.ndarray:
    masked = np.ma.masked_invalid(np.asarray(array, dtype=np.float32))
    vmin, vmax = ZScaleInterval(contrast=ZCONTRAST).get_limits(masked)
    values = np.asarray(masked.filled(vmax), dtype=float)
    scaled = (values - vmin) / (vmax - vmin)
    np.clip(scaled, 0.0, 1.0, out=scaled)
    return np.rint(scaled * 255).astype(np.uint8)


def jpeg_bytes(array: np.ndarray) -> np.ndarray:
    with io.BytesIO() as buffer:
        Image.fromarray(array).save(buffer, format="JPEG", quality=JPEG_QUALITY)
        return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def build_hdf5(
    metadata: pd.DataFrame,
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    n = len(metadata)
    groups = list(metadata.groupby("fits_path", sort=False))
    with h5py.File(output, "w") as h5:
        h5.create_dataset(
            "images", shape=(n,), dtype=h5py.vlen_dtype(np.dtype("uint8")),
            chunks=(3488,), compression="gzip", compression_opts=4,
        )
        h5.create_dataset("filenames", shape=(n,), dtype=h5py.string_dtype("utf-8"))
        for group_index, (fits_path, group) in enumerate(groups, 1):
            filename = Path(str(fits_path)).name
            local_path = Path(str(fits_path))
            print(f"FITS {group_index}/{len(groups)}: {filename} ({len(group):,} cutouts)")
            local_path.parent.mkdir(parents=True, exist_ok=True)
            Observations.download_file(f"mast:HST/product/{filename}", local_path=str(local_path), cache=False)
            with fits.open(local_path, memmap=True) as hdul:
                hdu_index = first_2d_hdu(hdul)
                data = hdul[hdu_index].data
                wcs = WCS(hdul[hdu_index].header)
                sky = SkyCoord(
                    group["SourceRA"].to_numpy(dtype=float),
                    group["SourceDec"].to_numpy(dtype=float),
                    unit="deg",
                    frame="icrs",
                )
                xs, ys = wcs.world_to_pixel(sky)
                for offset, row in enumerate(group.itertuples(index=False)):
                    out_index = int(row.shard_index)
                    cutout = Cutout2D(
                        data, position=(float(xs[offset]), float(ys[offset])),
                        size=(SIZE, SIZE), mode="partial", fill_value=np.nan, copy=True,
                    )
                    h5["images"][out_index] = jpeg_bytes(stretch_to_uint8(cutout.data))
                    h5["filenames"][out_index] = str(int(row.SourceID))
                del data, wcs
            gc.collect()
            local_path.unlink()
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", default="work")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--batch-rows", type=int, default=1_500_000)
    parser.add_argument("--quick-page-rows", type=int, default=2_000)
    parser.add_argument("--stop-after", choices=["catalog", "parquet", "all"], default="all")
    args = parser.parse_args()
    work = Path(args.work_dir)
    output_dir = Path(args.output_dir)
    work.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{STEM}.parquet"
    hdf5_path = output_dir / f"{STEM}.hdf5"

    inventory = mast_inventory(work)
    raw = work / "hsc_acswfc_f814w_extended_raw.parquet"
    load_dotenv(args.env_file, override=False, interpolate=False)
    cj = MastCasJobs(userid=int(os.environ["CASJOBS_WSID"]), password=os.environ["CASJOBS_PW"],
                     context="HSCv3", base_url=CASJOBS, request_type="POST")
    tag = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    table_prefix = f"ghs{tag}"
    image_table = f"{table_prefix}_images"
    image_names = inventory["image_name"].rename("ImageName").to_csv(index_label="Seq")
    cj.upload_table(image_table, image_names, exists=False)
    counts = query_counts(cj, inventory, image_table, f"{table_prefix}_counts")
    build_raw_catalog(
        cj, counts, work, image_table, table_prefix,
        args.batch_rows, args.quick_page_rows,
    )
    if args.stop_after == "catalog":
        return
    metadata = deduplicate(raw, inventory, parquet_path)
    if args.stop_after == "parquet":
        return
    build_hdf5(metadata, hdf5_path)


if __name__ == "__main__":
    main()
