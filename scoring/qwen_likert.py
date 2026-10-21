"""Score images with Qwen3.5-397B-A17B using the Likert method (Table 1, Figure 1).

How it works
  Same method and prompt as scoring/gemini_likert.py: 4x4 grids, each picked image is rated 1-5,
  every image appears in 10 grids, and its score is the sum (0 to 50). Qwen is called through
  OpenRouter with reasoning turned off and temperature 0.1.

  python3 scoring/qwen_likert.py

  You need OPENROUTER_API_KEY in your environment.

Test mode (TEST_MODE, top of the file)
  True (the default, and what the paper uses): score the 20,000-image benchmark. That is every
    cutout matched to one of the 167 reference anomalies, plus random background cutouts to make
    20,000 in total (TEST_NUM_IMAGES). The background draw uses the run's seed, so seed 44 gives
    exactly the images in paper_results/benchmark_subset.csv; other seeds draw a different
    background but always keep all 167 anomalies. Scores go to generated/scores/.
  False: score every cutout in the archive (223,195 of them), e.g. to search for new candidates.
    This costs about 11 times as much as test mode. Scores go to generated/scores_full_archive/.
    The reference anomalies are still flagged in the output, so you can still measure recall.

What you need
  - The image archive from dataset/pipeline.py (the .hdf5 cutouts and the .parquet positions),
    in ../hubble_data/ or wherever HUBBLE_DATA_DIR points.
  - dataset/reference_anomalies.csv (ships with the repo). Each entry labels its nearest cutout
    within 3 arcsec (MATCH_RADIUS_ARCSEC), which gives the 167 reference anomalies.
  - The DejaVu Sans Bold font at FONT_PATH. The default is the usual Linux location; on macOS,
    `brew install --cask font-dejavu` and set FONT_PATH to ~/Library/Fonts/DejaVuSans-Bold.ttf.
    The font draws the image numbers on each grid, so keep it if you want the paper's images.

Things you might want to change
  OPENROUTER_MODEL  any vision model on OpenRouter that supports structured outputs.
  RANDOM_SEEDS      the paper used seed 44 only.
  NUM_CORES         how many API requests run in parallel. Lower it if you hit rate limits.
  "provider"        in the request below; the paper's run skipped one OpenRouter provider
                    (DigitalOcean). Remove that line to let OpenRouter pick freely.

What it costs
  About $13 in test mode (OpenRouter's charge for the paper's run). OpenRouter reports the cost
  of each request, and the total is written at the bottom of the output file.

What you get
  generated/scores/qwen_likert_seed44.csv, sorted by score, plus a usage footer.
  Compare with paper_results/scores/qwen_likert_seed44.csv.

What to expect
  In the paper, the 1,984 images scoring 36 or more held 160 of the 167 anomalies (95.8%).
  OpenRouter can serve the model from different providers, so runs vary a bit more than Gemini's.
"""

import os
from pathlib import Path
import csv
import random
import math
import io
import base64
from multiprocessing import Pool

import h5py
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from openai import OpenAI
from pydantic import BaseModel, Field
from PIL import Image, ImageDraw, ImageFont
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.paths import (
    HDF5_PATH,
    INTERESTING_CSV_PATH,
    PARQUET_PATH,
    SUBSET_DIR,
    FULL_RUN_DIR,
)

OPENROUTER_MODEL = "qwen/qwen3.5-397b-a17b"

NUM_CORES = 25
ROUNDS_PER_IMAGE = 10
GRID_DIM = 4
BATCH_SIZE = GRID_DIM * GRID_DIM

MATCH_RADIUS_ARCSEC = 3.0  # radius search: match each Interesting.csv item to its nearest image
RANDOM_SEEDS = [44]

TEST_MODE = True  # True: the 20,000-image benchmark; False: the whole archive (see top)
TEST_NUM_IMAGES = 20000
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # grid numbers (see top)

class SelectedImage(BaseModel):
    GridIndex: int = Field(
        ge=1,
        le=16,
        description=(
            "Index 1-16 of the interesting image in the 4x4 grid, "
            "reading top-to-bottom, left-to-right."
        ),
    )
    Score: int = Field(
        ge=1,
        le=5,
        description=(
            "Scientific interest score from 1 mildly interesting "
            "to 5 exceptionally interesting."
        ),
    )

class SelectedImages(BaseModel):
    selections: list[SelectedImage] = Field(
        description="List of selected images. Empty list if none are interesting."
    )

SYSTEM_INSTRUCTION = (
    "You are an expert astronomer evaluating a 4x4 grid of astronomical images. "
    "The images are indexed 1-4 in the top row, 5-8 in the second row, "
    "9-12 in the third row, and 13-16 in the bottom row. "
    "Your task:\n"
    "1. Select any images that are scientifically interesting. "
    "Skip blank, empty, or uninformative frames.\n"
    "2. For each selected image, assign a Score from 1 to 5 "
    "(no explanation needed):\n"
    "   1 = mildly interesting, 2 = somewhat interesting, 3 = interesting, "
    "   4 = very interesting, 5 = exceptionally interesting.\n"
    "Return a list of selected images. If none are interesting, return an empty list.\n"
)


def load_interesting_metadata(csv_path, coordinates, radius_arcsec=MATCH_RADIUS_ARCSEC):
    import pandas as pd

    df = pd.read_csv(csv_path)


    catalog_ids = list(coordinates.keys())
    catalog_ra = np.array([coordinates[sid]["SourceRA"] for sid in catalog_ids])
    catalog_dec = np.array([coordinates[sid]["SourceDec"] for sid in catalog_ids])
    catalog_coords = SkyCoord(ra=catalog_ra * u.deg, dec=catalog_dec * u.deg)

    interesting_coords = SkyCoord(ra=df["RA"].values * u.deg, dec=df["Dec"].values * u.deg)

    # Each reference labels only its nearest image within the matching radius.
    nearest_idx, sep2d, _ = interesting_coords.match_to_catalog_sky(catalog_coords)

    metadata = {}
    n_items_matched = 0

    for row_idx, (match_idx, sep) in enumerate(zip(nearest_idx, sep2d.arcsec)):
        if sep <= radius_arcsec:
            n_items_matched += 1
            matched_source_id = catalog_ids[match_idx]
            classification = str(df.iloc[row_idx]["Classification"]).strip()
            metadata[matched_source_id] = classification

    print(
        f"Matched {n_items_matched} / {len(df)} Interesting.csv entries to an "
        f"image within {radius_arcsec}\" ({len(metadata)} unique images, "
        f"{n_items_matched - len(metadata)} shared their nearest image with "
        f"another Interesting.csv entry)."
    )

    return metadata


def load_parquet_coordinates(parquet_path):
    import pandas as pd

    df = pd.read_parquet(parquet_path)
    coordinates = {}

    for _, row in df.iterrows():
        source_id = str(int(row["SourceID"]))
        coordinates[source_id] = {
            "SourceRA": row["SourceRA"],
            "SourceDec": row["SourceDec"],
        }

    print(f"Loaded coordinates for {len(coordinates)} sources from parquet.")
    return coordinates


def build_labeled_records(hdf5_path, interesting_csv_path, coordinates):
    print("Loading all images from HDF5...")

    interesting_metadata = load_interesting_metadata(interesting_csv_path, coordinates)

    with h5py.File(hdf5_path, "r") as f:
        filenames = f["filenames"].asstr()
        images = f["images"]

        n_total = len(images)
        all_records = []
        for hdf5_idx in range(n_total):
            filename = filenames[hdf5_idx]
            img_bytes = bytes(images[hdf5_idx])

            source_id = os.path.splitext(filename)[0]
            classification = interesting_metadata.get(source_id, "")
            is_interesting = 1 if classification else 0
            coord_data = coordinates[source_id]

            all_records.append({
                "index": hdf5_idx,
                "filename": filename,
                "image_bytes": img_bytes,
                "interesting": is_interesting,
                "classification": classification,
                "SourceRA": coord_data["SourceRA"],
                "SourceDec": coord_data["SourceDec"],
            })

    n_interesting_found = sum(r["interesting"] for r in all_records)

    print(f"HDF5 contains {n_total} images.")
    print(f"Found {n_interesting_found} interesting images in CSV.")

    return all_records


def prepare_run_records(all_records, random_seed):
    # Sample backgrounds, then shuffle with the same per-seed RNG.
    rng = random.Random(random_seed)
    records = [dict(r) for r in all_records]

    if TEST_MODE:
        interesting = [r for r in records if r["interesting"]]
        boring = [r for r in records if not r["interesting"]]

        n_boring_needed = max(0, TEST_NUM_IMAGES - len(interesting))
        n_boring_needed = min(n_boring_needed, len(boring))
        boring_subset = rng.sample(boring, n_boring_needed)

        records = interesting + boring_subset

    rng.shuffle(records)

    for i, record in enumerate(records):
        record["index"] = i

    print(f"Seed {random_seed}: {len(records)} images, "
          f"{sum(record['interesting'] for record in records)} reference anomalies")

    return records


def create_grid(image_records):
    thumb_w, thumb_h = 200, 200
    margin = 4

    grid_w = (thumb_w * GRID_DIM) + (margin * (GRID_DIM + 1))
    grid_h = (thumb_h * GRID_DIM) + (margin * (GRID_DIM + 1))

    canvas = Image.new("RGB", (grid_w, grid_h), color=(50, 50, 50))
    draw = ImageDraw.Draw(canvas)

    font = ImageFont.truetype(
        FONT_PATH,
        20,
    )

    for i, record in enumerate(image_records):
        if record is None:
            continue

        row = i // GRID_DIM
        col = i % GRID_DIM

        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + margin)

        img_bytes = record["image_bytes"]

        with Image.open(io.BytesIO(img_bytes)) as img:
            img = img.convert("RGB")
            img = img.resize((thumb_w, thumb_h))
            canvas.paste(img, (x, y))

        draw.text(
            (x + 5, y + 5),
            str(i + 1),
            fill="white",
            font=font,
            stroke_width=1,
            stroke_fill="black",
        )

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)

    return buf


def process_batches(args):
    batches, api_key = args
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    decisions = []
    token_totals = [0] * 3
    for batch_records in batches:
        image_bytes = create_grid(batch_records).getvalue()
        grid_b64 = base64.b64encode(image_bytes).decode()
        response = client.beta.chat.completions.parse(
            model=OPENROUTER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this 4x4 grid of astronomical images. "
                            "Select any scientifically interesting images and "
                            "rate each selected image from 1 to 5."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{grid_b64}"
                        },
                    },
                ]},
            ],
            response_format=SelectedImages,
            temperature=0.1,
            extra_body={
                "usage": {"include": True},
                "reasoning": {"enabled": False},
                "provider": {"ignore": ["DigitalOcean"]},
            },
        )
        usage = response.usage
        request_tokens = [usage.prompt_tokens, usage.completion_tokens, usage.cost]
        for i, value in enumerate(request_tokens):
            token_totals[i] += value
        data = response.choices[0].message.parsed.selections
        # Keep repeated score entries: the original runs sum every returned entry.
        for item in data:
            record = batch_records[item.GridIndex - 1]
            if record is not None:
                decisions.append((record["index"], item.Score))
    return decisions, token_totals


def run_experiment(all_images, api_key, random_seed, results_file):
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    scores = {record["index"]: 0 for record in all_images}
    token_totals = [0] * 3
    random.seed(random_seed)
    # Each image receives ten exposures; a grid can contain repeated images.
    master_pool = all_images * ROUNDS_PER_IMAGE
    random.shuffle(master_pool)
    batches = []
    for start in range(0, len(master_pool), BATCH_SIZE):
        batch = master_pool[start:start + BATCH_SIZE]
        batches.append(batch + [None] * (BATCH_SIZE - len(batch)))
    chunk_size = math.ceil(len(batches) / NUM_CORES)
    workers = [(batches[start:start + chunk_size], api_key)
               for start in range(0, len(batches), chunk_size)]
    with Pool(processes=NUM_CORES) as pool:
        for decisions, usage in pool.map(process_batches, workers):
            for index, score in decisions:
                scores[index] += score
            for i, value in enumerate(usage):
                token_totals[i] += value

    records = {record["index"]: record for record in all_images}
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'filename', 'imagescore', 'interesting', 'classification', 'SourceRA', 'SourceDec'])
        for index in sorted(scores, key=scores.get, reverse=True):
            record = records[index]
            writer.writerow([index, record["filename"], scores[index], record["interesting"],
                             record["classification"], record["SourceRA"], record["SourceDec"]])
        writer.writerow([])
        writer.writerow(["# TOKEN USAGE SUMMARY"])
        writer.writerow(["# TotalInputTokens", token_totals[0]])
        writer.writerow(["# TotalOutputTokens", token_totals[1]])
        writer.writerow(["# TotalTokens", sum(token_totals[:2])])
        writer.writerow(["# TotalCostUSD", token_totals[2]])
    print(f"Wrote {results_file}")


if __name__ == "__main__":
    api_key = os.environ["OPENROUTER_API_KEY"]
    coordinates = load_parquet_coordinates(PARQUET_PATH)
    all_labeled = build_labeled_records(HDF5_PATH, INTERESTING_CSV_PATH, coordinates)
    output_dir = Path(SUBSET_DIR if TEST_MODE else FULL_RUN_DIR)
    for seed in RANDOM_SEEDS:
        run_images = prepare_run_records(all_labeled, seed)
        run_experiment(run_images, api_key, seed, output_dir / f"qwen_likert_seed{seed}.csv")
