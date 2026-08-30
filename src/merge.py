import pandas as pd

nature = pd.read_csv('data/annotated/nature_colony_labels.csv')
nature['source_dataset'] = 'nature_colony'

supplemental = pd.read_csv('data/annotated/nature_colony_labels_supplementary.csv')
supplemental['source_dataset'] = '22022540'

merged = pd.concat([nature, supplemental], ignore_index=True)
merged.to_csv('data/annotated/merged_labels.csv', index=False)

print(merged.groupby(['genus', 'source_dataset']).size())