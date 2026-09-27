"""
Build a supplementary classifier CSV for the 22022540 dataset.

This dataset is organized as flattened YOLO files (e.g. sp04_img01.jpg +
sp04_img01.txt), but the classifier layer expects the legacy species labels
already used by the Nature colony dataset. We therefore:

1) keep only the six species that already exist in the original dataset,
2) map their new prefixes (sp04, sp11, sp13, sp20, sp21, sp23) to the
   legacy species codes (bce, eco, kpn, ses, sau, sag), and
3) export a CSV with the same schema used by the downstream classifier.
"""

import csv
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

BASE = ROOT_DIR / "data" / "raw" / "22022540"
IMAGES_DIR = BASE / "images"
LABELS_DIR = BASE / "label" / "annot_YOLO"
OUTPUT_CSV = ROOT_DIR / "data" / "annotated" / "nature_colony_labels_supplementary.csv"


def csv_filename(path):
    return path.relative_to(ROOT_DIR).as_posix()

# New dataset species prefix -> legacy species metadata that matches the
# classifier's original genus labels.
SPECIES_MAP = {
    "sp04": {"species_code": "bce", "genus": "bacillus_cereus", "gram": "positive", "agar": "BAP"},
    "sp11": {"species_code": "eco", "genus": "escherichia_coli", "gram": "negative", "agar": "Nutrient"},
    "sp13": {"species_code": "kpn", "genus": "klebsiella_pneumoniae", "gram": "negative", "agar": "BAP"},
    "sp20": {"species_code": "ses", "genus": "salmonella_enterica", "gram": "negative", "agar": "Nutrient"},
    "sp21": {"species_code": "sau", "genus": "staphylococcus_aureus", "gram": "positive", "agar": "BAP"},
    "sp23": {"species_code": "sag", "genus": "streptococcus_agalactiae", "gram": "positive", "agar": "BAP"},
}

rows = []
missing_labels = 0
missing_images = 0

if not IMAGES_DIR.exists():
    raise FileNotFoundError(f"Images directory not found: {IMAGES_DIR}")

if not LABELS_DIR.exists():
    raise FileNotFoundError(f"YOLO label directory not found: {LABELS_DIR}")

for image_file in sorted(IMAGES_DIR.glob("*.jpg")):
    stem = image_file.stem
    species_prefix = stem.rsplit("_", 1)[0]
    species_info = SPECIES_MAP.get(species_prefix)

    if species_info is None:
        continue

    legacy_code = species_info["species_code"]
    label_file = LABELS_DIR / f"{stem}.txt"

    if not label_file.exists():
        missing_images += 1
        continue

    with open(label_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    num_colonies = len(lines)
    rows.append(
        {
            "filename": csv_filename(image_file),
            "label_file": label_file.name,
            "species_code": legacy_code,
            "genus": species_info["genus"],
            "gram_stain": species_info["gram"],
            "agar_type": species_info["agar"],
            "imaging_modality": "colony_photo",
            "stain_type": "not_applicable",
            "num_colony_instances": num_colonies,
            "source": "nature_colony_supplementary",
            "source_id": f"nature_colony_supplementary_{legacy_code}",
            "split": "",
        }
    )

# Some datasets may use uppercase or JPEG files; include them too.
for image_file in sorted(IMAGES_DIR.glob("*.jpeg")):
    if any(row["filename"] == csv_filename(image_file) for row in rows):
        continue
    stem = image_file.stem
    species_prefix = stem.rsplit("_", 1)[0]
    species_info = SPECIES_MAP.get(species_prefix)
    if species_info is None:
        continue

    label_file = LABELS_DIR / f"{stem}.txt"
    if not label_file.exists():
        missing_images += 1
        continue

    with open(label_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    rows.append(
        {
            "filename": csv_filename(image_file),
            "label_file": label_file.name,
            "species_code": species_info["species_code"],
            "genus": species_info["genus"],
            "gram_stain": species_info["gram"],
            "agar_type": species_info["agar"],
            "imaging_modality": "colony_photo",
            "stain_type": "not_applicable",
            "num_colony_instances": len(lines),
            "source": "nature_colony_supplementary",
            "source_id": f"nature_colony_supplementary_{species_info['species_code']}",
            "split": "",
        }
    )

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

if rows:
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
else:
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "filename",
                "label_file",
                "species_code",
                "genus",
                "gram_stain",
                "agar_type",
                "imaging_modality",
                "stain_type",
                "num_colony_instances",
                "source",
                "source_id",
                "split",
            ]
        )

print(f"{'=' * 60}")
print(f"Total images matched: {len(rows)}")
print(f"Species included: {len(set(r['species_code'] for r in rows))}")
print(f"Images without matching label file: {missing_images}")
print(f"Output written to: {OUTPUT_CSV}")

counts = Counter(r["species_code"] for r in rows)
print("\nImages per species:")
for code, count in sorted(counts.items()):
    print(f"  {code}: {count}")

print(f"\nTotal colony instances across all images: {sum(r['num_colony_instances'] for r in rows)}")
