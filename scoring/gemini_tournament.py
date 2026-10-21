"""Score images with Gemini 3.1 Flash-Lite using the Tournament method (Table 1, Figure 1).

How it works
  Each round, the remaining images are shuffled into 2x2 grids and Gemini decides, for each of
  the four, whether it's interesting enough to keep for a human to look at (any number can be
  kept). Kept images go through to the next round. After 10 rounds (MAX_ROUNDS), an image's
  score is the number of rounds it survived: 0 to 10. When the images don't split evenly into
  fours, the leftovers go through automatically. Temperature is 0.1 and thinking is "low".

  python3 scoring/gemini_tournament.py

  You need GEMINI_API_KEY in your environment.

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
  RANDOM_SEEDS   one run per seed. The paper reports seed 44 only.
  MAX_ROUNDS     more rounds give a shorter final list.
  NUM_CORES      how many API requests run in parallel. Lower it if you hit rate limits.

What it costs
  About $7 per seed in test mode at the paper's prices. Round 1 sends all 20,000 images (5,000
  grids) and later rounds only the survivors. Token counts are written at the bottom of the
  output file.

What you get
  generated/scores/gemini_tournament_seed{seed}.csv, sorted by score, plus a token-usage footer.
  Compare with paper_results/scores/gemini_tournament_seed44.csv.

What to expect
  In the paper, 2,019 images survived all 10 rounds and held 160 of the 167 anomalies (95.8%).
  Model outputs vary a little from run to run, so expect numbers close to these, not identical.
"""

import os
from pathlib import Path
import csv
import random
import math
import io
from multiprocessing import Pool

import h5py
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from google import genai
from google.genai import types
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

NUM_CORES = 25

GROUP_SIZE = 4  # 2x2 grid of images evaluated per API call
QUAD_THUMB_W = 240
QUAD_THUMB_H = 240

MATCH_RADIUS_ARCSEC = 3.0  # radius search: match each Interesting.csv item to its nearest image

MAX_ROUNDS = 10
RANDOM_SEEDS = [44]

TEST_MODE = True  # True: the 20,000-image benchmark; False: the whole archive (see top)
TEST_NUM_IMAGES = 20000
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # grid numbers (see top)

class InterestingSelection(BaseModel):
    KeepImage1: bool = Field(
        description=(
            "True if Image 1 (top-left) is scientifically interesting enough "
            "to keep for human inspection. False otherwise."
        )
    )
    KeepImage2: bool = Field(
        description=(
            "True if Image 2 (top-right) is scientifically interesting enough "
            "to keep for human inspection. False otherwise."
        )
    )
    KeepImage3: bool = Field(
        description=(
            "True if Image 3 (bottom-left) is scientifically interesting enough "
            "to keep for human inspection. False otherwise."
        )
    )
    KeepImage4: bool = Field(
        description=(
            "True if Image 4 (bottom-right) is scientifically interesting enough "
            "to keep for human inspection. False otherwise."
        )
    )

config = types.GenerateContentConfig(
    system_instruction=(
        "You are an expert astronomer evaluating a 2x2 grid of four astronomical "
        "images, labeled Image 1 (top-left), Image 2 (top-right), Image 3 "
        "(bottom-left), and Image 4 (bottom-right). "
        "Evaluate each image independently. Do not force a winner. "
        "For each image, decide whether it is scientifically interesting enough "
        "to keep for human inspection. "
        "It is acceptable to keep any number of the four images (none, some, or all). "
        "Interesting images may show unusual morphology, asymmetry, interactions, "
        "merger-like features, arcs, rings, tails, shells, clumps, distortions, "
        "rare-looking objects, or other features worth human inspection. "
        "Reject artifacts, blank images, noisy frames, and obvious non-astronomical defects. "
        "Return only whether to keep each of Image 1, Image 2, Image 3, and Image 4. "
        "No explanation needed."
    ),
    response_mime_type="application/json",
    response_schema=InterestingSelection,
    temperature=0.1,
    thinking_config=types.ThinkingConfig(thinking_level="low"),
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


def create_quad_image(quad_records):
    margin = 8
    label_h = 32
    cell_w = QUAD_THUMB_W
    cell_h = QUAD_THUMB_H + label_h

    canvas_w = cell_w * 2 + margin * 3
    canvas_h = cell_h * 2 + margin * 3

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(50, 50, 50))
    draw = ImageDraw.Draw(canvas)

    font = ImageFont.truetype(
        FONT_PATH,
        22,
    )

    for i, record in enumerate(quad_records):
        if record is None:
            continue

        row = i // 2
        col = i % 2

        x = margin + col * (cell_w + margin)
        y = margin + row * (cell_h + margin)

        label = f"Image {i + 1}"
        draw.text(
            (x + 6, y),
            label,
            fill="white",
            font=font,
            stroke_width=1,
            stroke_fill="black",
        )

        img_bytes = record["image_bytes"]

        with Image.open(io.BytesIO(img_bytes)) as img:
            img = img.convert("RGB").resize((QUAD_THUMB_W, QUAD_THUMB_H))
            canvas.paste(img, (x, y + label_h))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf


def split_into_groups(active_records, group_size=GROUP_SIZE):
    active_records = active_records[:]
    bye_records = []

    n_bye = len(active_records) % group_size
    for _ in range(n_bye):
        bye_records.append(active_records.pop())

    groups = []
    for i in range(0, len(active_records), group_size):
        group_number = len(groups) + 1
        groups.append((group_number, active_records[i:i + group_size]))

    return groups, bye_records


def process_matches(args):
    groups, api_key = args
    client = genai.Client(api_key=api_key)
    decisions = []
    token_totals = [0] * 4
    for group_number, group_records in groups:
        image_bytes = create_quad_image(group_records).getvalue()
        quad_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        contents = [
            (
                "Evaluate this 2x2 grid of four astronomical images. "
                "Image 1 is top-left, Image 2 is top-right, Image 3 is "
                "bottom-left, and Image 4 is bottom-right. "
                "Evaluate each image independently. Do not force a winner. "
                "For each image, decide whether it is scientifically interesting "
                "enough to keep for human inspection."
            ),
            quad_part,
        ]
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite",
            contents=contents,
            config=config,
        )
        usage = response.usage_metadata
        request_tokens = [usage.prompt_token_count, usage.candidates_token_count,
                          usage.thoughts_token_count or 0, usage.cached_content_token_count or 0]
        for i, value in enumerate(request_tokens):
            token_totals[i] += value
        data = response.parsed
        keep_flags = [data.KeepImage1, data.KeepImage2, data.KeepImage3, data.KeepImage4]
        decisions.extend((record["index"], int(keep))
                         for record, keep in zip(group_records, keep_flags))
    return decisions, token_totals


def run_experiment(all_images, api_key, random_seed, results_file):
    Path(results_file).parent.mkdir(parents=True, exist_ok=True)
    scores = {record["index"]: 0 for record in all_images}
    token_totals = [0] * 4
    random.seed(random_seed)
    active_records = all_images[:]
    for round_number in range(1, MAX_ROUNDS + 1):
        random.shuffle(active_records)
        groups, bye_records = split_into_groups(active_records)
        decisions = {}
        if groups:
            chunk_size = math.ceil(len(groups) / NUM_CORES)
            workers = [(groups[start:start + chunk_size], api_key)
                       for start in range(0, len(groups), chunk_size)]
            with Pool(processes=NUM_CORES) as pool:
                for choices, usage in pool.map(process_matches, workers):
                    decisions.update(choices)
                    for i, value in enumerate(usage):
                        token_totals[i] += value
        # Leftovers advance automatically and receive the same point as model-kept images.
        decisions.update((record["index"], 1) for record in bye_records)
        active_records = [record for record in active_records if decisions[record["index"]]]
        for record in active_records:
            scores[record["index"]] += 1
        print(f"Round {round_number}: {len(active_records)} images retained")
        if not active_records:
            break

    records = {record["index"]: record for record in all_images}
    with open(results_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'ImageScore', 'interesting', 'classification', 'SourceRA', 'SourceDec'])
        for index in sorted(scores, key=scores.get, reverse=True):
            record = records[index]
            writer.writerow([record["filename"], scores[index], record["interesting"],
                             record["classification"], record["SourceRA"], record["SourceDec"]])
        writer.writerow([])
        writer.writerow(["# TOKEN USAGE SUMMARY"])
        writer.writerow(["# TotalInputTokens", token_totals[0]])
        writer.writerow(["# TotalOutputTokens", token_totals[1]])
        writer.writerow(["# TotalThinkingTokens", token_totals[2]])
        writer.writerow(["# TotalCachedInputTokens", token_totals[3]])
        writer.writerow(["# TotalTokens", sum(token_totals[:3])])
    print(f"Wrote {results_file}")


if __name__ == "__main__":
    api_key = os.environ["GEMINI_API_KEY"]
    coordinates = load_parquet_coordinates(PARQUET_PATH)
    all_labeled = build_labeled_records(HDF5_PATH, INTERESTING_CSV_PATH, coordinates)
    output_dir = Path(SUBSET_DIR if TEST_MODE else FULL_RUN_DIR)
    for seed in RANDOM_SEEDS:
        run_images = prepare_run_records(all_labeled, seed)
        run_experiment(run_images, api_key, seed, output_dir / f"gemini_tournament_seed{seed}.csv")
