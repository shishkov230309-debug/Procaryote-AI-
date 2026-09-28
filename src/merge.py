import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ANNOTATED_DIR = ROOT_DIR / "data" / "annotated"
OUTPUT_CSV = ANNOTATED_DIR / "merged_labels.csv"


def _duplicate_values(series, basename=False):
	values = series.astype(str)
	if basename:
		values = values.map(lambda value: Path(value).name.casefold())
	return values[values.duplicated(keep=False)].unique().tolist()


def _validate_merged_rows(dataframe):
	for column in ("filename", "genus"):
		if column not in dataframe.columns:
			raise ValueError(f"Merged labels are missing required column: {column}")

	duplicate_filenames = _duplicate_values(dataframe["filename"])
	if duplicate_filenames:
		raise ValueError(
			f"Merged labels contain duplicate filenames: {duplicate_filenames[:10]}"
		)

	duplicate_basenames = _duplicate_values(dataframe["filename"], basename=True)
	if duplicate_basenames:
		raise ValueError(
			f"Merged labels contain duplicate basenames: {duplicate_basenames[:10]}"
		)


def main():
	nature = pd.read_csv(ANNOTATED_DIR / "nature_colony_labels.csv")
	nature["source_dataset"] = "nature_colony"

	supplemental = pd.read_csv(
		ANNOTATED_DIR / "nature_colony_labels_supplementary.csv"
	)
	supplemental["source_dataset"] = "22022540"

	merged = pd.concat([nature, supplemental], ignore_index=True)
	_validate_merged_rows(merged)
	merged.to_csv(OUTPUT_CSV, index=False)

	print(f"Merged {len(merged)} images across {merged['genus'].nunique()} classes")
	print("\nSource distribution:")
	print(merged["source_dataset"].value_counts().sort_index().to_string())
	print("\nClass/source distribution:")
	print(merged.groupby(["genus", "source_dataset"]).size().to_string())
	print(f"\nOutput written to: {OUTPUT_CSV}")


if __name__ == "__main__":
	main()