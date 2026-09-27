import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ANNOTATED_DIR = ROOT_DIR / "data" / "annotated"

nature = pd.read_csv(ANNOTATED_DIR / "nature_colony_labels.csv")
nature['source_dataset'] = 'nature_colony'

supplemental = pd.read_csv(ANNOTATED_DIR / "nature_colony_labels_supplementary.csv")
supplemental['source_dataset'] = '22022540'

merged = pd.concat([nature, supplemental], ignore_index=True)
merged.to_csv(ANNOTATED_DIR / "merged_labels.csv", index=False)

print(merged.groupby(['genus', 'source_dataset']).size())