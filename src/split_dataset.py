"""
Split nature_colony_labels.csv into train/val/test sets.

Stratified split: each of the 19 species gets the same proportion
in train/val/test, so no split ends up missing a species or
overloaded with one.

Split ratios: 70% train / 15% val / 15% test
With 50 images per species, that's roughly 35/7/8 per species.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_CSV = "data/annotated/nature_colony_labels.csv"
OUTPUT_CSV = "data/annotated/nature_colony_labels_split.csv"

df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} images across {df['genus'].nunique()} species")

# First split: separate out the test set (15%)
train_val_df, test_df = train_test_split(
    df,
    test_size=0.15,
    stratify=df["genus"],   # keeps species proportions equal in both halves
    random_state=42          # makes the split reproducible every time you run this
)

# Second split: divide the remaining 85% into train (70% of total) and val (15% of total)
# 0.15 / 0.85 ≈ 0.1765 to get val = 15% of the original total
train_df, val_df = train_test_split(
    train_val_df,
    test_size=0.1765,
    stratify=train_val_df["genus"],
    random_state=42
)

# Tag each row with its split
train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

# Recombine into one file
final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
final_df.to_csv(OUTPUT_CSV, index=False)

# Report
print(f"\n{'='*50}")
print(f"Train: {len(train_df)} images")
print(f"Val:   {len(val_df)} images")
print(f"Test:  {len(test_df)} images")
print(f"\nOutput written to: {OUTPUT_CSV}")

print(f"\nPer-species breakdown:")
breakdown = final_df.groupby(["genus", "split"]).size().unstack(fill_value=0)
print(breakdown)
