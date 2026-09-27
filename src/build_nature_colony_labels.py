"""
Build labels_master.csv from the nature_colony dataset.

Scans data/raw/nature_colony/images/images/<species_code>/
and matches each image to its YOLO label file in
data/raw/nature_colony/label/label/YOLO_txt/<species_code>_txt/

Species mapping verified against Table 1 of the source paper:
https://www.nature.com/articles/s41597-026-07095-5/tables/1
"""

import os
import csv
from pathlib import Path
from collections import Counter

ROOT_DIR = Path(__file__).resolve().parent.parent
BASE = ROOT_DIR / "data" / "raw" / "nature_colony"
IMAGES_DIR = BASE / "images" / "images"
LABELS_DIR = BASE / "label" / "label" / "YOLO_txt"
OUTPUT_CSV = ROOT_DIR / "data" / "annotated" / "nature_colony_labels.csv"

# Verified from Table 1 of the Nature paper
SPECIES_MAP = {
    "aba": {"genus": "acinetobacter_baumannii", "gram": "negative", "agar": "MAC"},
    "eco": {"genus": "escherichia_coli", "gram": "negative", "agar": "BAP"},
    "kpn": {"genus": "klebsiella_pneumoniae", "gram": "negative", "agar": "MAC"},
    "sau": {"genus": "staphylococcus_aureus", "gram": "positive", "agar": "BAP"},
    "ses": {"genus": "salmonella_enterica", "gram": "negative", "agar": "SS"},
    "sma": {"genus": "serratia_marcescens", "gram": "negative", "agar": "MAC"},
    "sep": {"genus": "staphylococcus_epidermidis", "gram": "positive", "agar": "BAP"},
    "efa": {"genus": "enterococcus_faecium", "gram": "positive", "agar": "BAP"},
    "stm": {"genus": "stenotrophomonas_maltophilia", "gram": "negative", "agar": "MAC"},
    "eca": {"genus": "enterococcus_casseliflavus", "gram": "positive", "agar": "BAP"},
    "enc": {"genus": "enterobacter_cloacae", "gram": "negative", "agar": "MAC"},
    "cst": {"genus": "corynebacterium_striatum", "gram": "positive", "agar": "BAP"},
    "mmo": {"genus": "morganella_morganii", "gram": "negative", "agar": "MAC"},
    "bce": {"genus": "bacillus_cereus", "gram": "positive", "agar": "BAP"},
    "spy": {"genus": "streptococcus_pyogenes", "gram": "positive", "agar": "BAP"},
    "sag": {"genus": "streptococcus_agalactiae", "gram": "positive", "agar": "BAP"},
    "ppu": {"genus": "pseudomonas_putida", "gram": "negative", "agar": "BAP"},
    "spn": {"genus": "streptococcus_pneumoniae", "gram": "positive", "agar": "BAP"},
    "bcp": {"genus": "burkholderia_cepacia", "gram": "negative", "agar": "MAC"},
}

rows = []
missing_labels = 0
missing_images = 0

for species_code in sorted(os.listdir(IMAGES_DIR)):
    img_folder = IMAGES_DIR / species_code
    label_folder = LABELS_DIR / f"{species_code}_txt"

    if not img_folder.is_dir():
        continue

    if not label_folder.is_dir():
        print(f"WARNING: no label folder for '{species_code}', skipping")
        missing_labels += 1
        continue

    species_info = SPECIES_MAP.get(species_code)
    if species_info is None:
        print(f"WARNING: '{species_code}' not in SPECIES_MAP, skipping")
        continue

    for img_file in sorted(img_folder.glob("*.jpg")):
        label_file = label_folder / (img_file.stem + ".txt")

        if not label_file.exists():
            missing_images += 1
            continue

        with open(label_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        num_colonies = len(lines)
        n = img_file.stem.split('_')[-1]  # Extract the number from the filename

        rows.append({
            "filename":f"data/raw/nature_colony/images/images/{species_code}/{species_code}_{n}.jpg",
            "label_file": f"{species_code}_txt/{label_file.name}",
            "species_code": species_code,
            "genus": species_info["genus"],
            "gram_stain": species_info["gram"],
            "agar_type": species_info["agar"],
            "imaging_modality": "colony_photo",
            "stain_type": "not_applicable",
            "num_colony_instances": num_colonies,
            "source": "nature_colony",
            "source_id": f"nature_colony_{species_code}",
            "split": ""
        })

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\n{'='*50}")
print(f"Total images matched: {len(rows)}")
print(f"Species folders processed: {len(set(r['species_code'] for r in rows))}")
print(f"Missing label folders: {missing_labels}")
print(f"Images without matching label file: {missing_images}")
print(f"Output written to: {OUTPUT_CSV}")

counts = Counter(r["species_code"] for r in rows)
print(f"\nImages per species:")
for code, count in sorted(counts.items()):
    genus = SPECIES_MAP[code]["genus"]
    print(f"  {code} ({genus}): {count}")

total_colonies = sum(r["num_colony_instances"] for r in rows)
print(f"\nTotal colony instances across all images: {total_colonies}")
