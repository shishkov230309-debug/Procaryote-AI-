"""Create a reproducible, leakage-checked train/val/test split."""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = ROOT_DIR / "data" / "annotated" / "merged_labels.csv"
OUTPUT_CSV = ROOT_DIR / "data" / "annotated" / "merged_labels_split.csv"
RANDOM_SEED = 42
SPLIT_NAMES = ("train", "val", "test")


def _basename_series(dataframe):
    return dataframe["filename"].astype(str).map(
        lambda filename: Path(filename).name.casefold()
    )


def _validate_unique_images(dataframe):
    duplicate_filenames = dataframe.loc[
        dataframe["filename"].duplicated(keep=False), "filename"
    ].unique()
    if len(duplicate_filenames):
        raise ValueError(
            f"Duplicate filenames would leak across the dataset: "
            f"{duplicate_filenames[:10].tolist()}"
        )

    basenames = _basename_series(dataframe)
    duplicate_basenames = basenames[basenames.duplicated(keep=False)].unique()
    if len(duplicate_basenames):
        raise ValueError(
            f"Duplicate basenames would leak across the dataset: "
            f"{duplicate_basenames[:10].tolist()}"
        )


def _validate_split(dataframe):
    if set(dataframe["split"]) != set(SPLIT_NAMES):
        raise ValueError(f"Expected exactly these split names: {SPLIT_NAMES}")

    for column in ("genus", "source_dataset"):
        if dataframe[column].isna().any() or dataframe[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Split output contains blank {column} values.")

    for genus, counts in dataframe.groupby("genus")["split"].value_counts().groupby(level=0):
        missing = set(SPLIT_NAMES) - set(counts.index.get_level_values("split"))
        if missing:
            raise ValueError(f"Class {genus!r} is missing splits: {sorted(missing)}")

    _validate_unique_images(dataframe)


def _report_distribution(dataframe):
    print("\nSource distribution:")
    print(dataframe.groupby(["source_dataset", "split"]).size().unstack(fill_value=0))
    print("\nClass distribution:")
    print(dataframe.groupby(["genus", "split"]).size().unstack(fill_value=0))
    print("\nClass/source distribution:")
    print(
        dataframe.groupby(["genus", "source_dataset", "split"])
        .size()
        .unstack(fill_value=0)
    )


def main():
    dataframe = pd.read_csv(INPUT_CSV, keep_default_na=False)
    required_columns = {"filename", "genus", "source_dataset"}
    missing_columns = required_columns - set(dataframe.columns)
    if missing_columns:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing_columns)}")
    if dataframe.empty:
        raise ValueError("Input CSV is empty.")

    _validate_unique_images(dataframe)
    dataframe = dataframe.sort_values("filename", kind="mergesort").reset_index(drop=True)
    dataframe["strat_key"] = (
        dataframe["genus"].astype(str) + "|" + dataframe["source_dataset"].astype(str)
    )

    print(f"Loaded {len(dataframe)} images across {dataframe['genus'].nunique()} classes")
    train_val_df, test_df = train_test_split(
        dataframe,
        test_size=0.15,
        stratify=dataframe["strat_key"],
        random_state=RANDOM_SEED,
    )
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=0.1765,
        stratify=train_val_df["strat_key"],
        random_state=RANDOM_SEED,
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    final_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    final_df = final_df.drop(columns=["strat_key"])
    _validate_split(final_df)
    final_df.to_csv(OUTPUT_CSV, index=False)

    print(f"\nTrain: {len(train_df)} images")
    print(f"Val:   {len(val_df)} images")
    print(f"Test:  {len(test_df)} images")
    print(f"\nOutput written to: {OUTPUT_CSV}")
    _report_distribution(final_df)


if __name__ == "__main__":
    main()
