from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import AutoImageProcessor

ROOT_DIR = Path(__file__).resolve().parent.parent
REQUIRED_COLUMNS = {"filename", "genus", "split"}
VALID_SPLITS = {"train", "val", "test"}


train_augment = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    ]
)


def _read_dataset_csv(csv_path):
    csv_path = Path(csv_path)
    try:
        df = pd.read_csv(csv_path, keep_default_na=False)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError(f"Could not read dataset CSV {csv_path}: {exc}") from exc

    if df.empty:
        raise ValueError(f"Dataset CSV {csv_path} is empty.")

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset CSV {csv_path} is malformed; missing columns: "
            f"{sorted(missing_columns)}."
        )

    for column in REQUIRED_COLUMNS:
        blank_rows = df[column].astype(str).str.strip() == ""
        if blank_rows.any():
            rows = (df.index[blank_rows] + 2).tolist()
            raise ValueError(
                f"Dataset CSV {csv_path} has blank {column!r} values at "
                f"CSV rows {rows[:10]}."
            )

    unknown_splits = set(df["split"]) - VALID_SPLITS
    if unknown_splits:
        raise ValueError(
            f"Dataset CSV {csv_path} contains unknown split values: "
            f"{sorted(unknown_splits)}."
        )

    if df["filename"].duplicated().any():
        duplicates = df.loc[df["filename"].duplicated(keep=False), "filename"].unique()
        raise ValueError(
            f"Dataset CSV {csv_path} contains duplicate filenames: "
            f"{duplicates[:10].tolist()}."
        )

    return df


def _resolve_image_path(filename):
    image_path = Path(filename)
    return image_path if image_path.is_absolute() else ROOT_DIR / image_path


def _validate_image_files(df, csv_path):
    missing = []
    invalid = []
    for row_number, filename in zip(df.index + 2, df["filename"]):
        image_path = _resolve_image_path(filename)
        if not image_path.is_file():
            missing.append((row_number, str(filename)))
            continue
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, SyntaxError) as exc:
            invalid.append((row_number, str(filename), str(exc)))

    if missing:
        raise FileNotFoundError(
            f"Dataset CSV {csv_path} references missing image files: "
            f"{missing[:10]}"
        )
    if invalid:
        raise ValueError(
            f"Dataset CSV {csv_path} references invalid image files: "
            f"{invalid[:10]}"
        )


def build_class_mapping(csv_path):
    df = _read_dataset_csv(csv_path)
    class_names = sorted(df["genus"].unique().tolist())
    return {
        "class_names": class_names,
        "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
        "num_classes": len(class_names),
    }


def validate_class_mapping(df, class_mapping):
    expected_names = set(class_mapping["class_names"])
    observed_names = set(df["genus"])
    if observed_names != expected_names:
        unknown_labels = sorted(observed_names - expected_names)
        missing_labels = sorted(expected_names - observed_names)
        details = []
        if unknown_labels:
            details.append(f"unknown labels {unknown_labels}")
        if missing_labels:
            details.append(f"missing labels {missing_labels}")
        raise ValueError(
            "Dataset labels do not exactly match the class mapping ("
            + "; ".join(details)
            + ")."
        )
    if int(class_mapping["num_classes"]) != len(expected_names):
        raise ValueError("Class mapping has an inconsistent number of classes.")
    expected_mapping = {
        name: index for index, name in enumerate(class_mapping["class_names"])
    }
    if class_mapping["class_to_idx"] != expected_mapping:
        raise ValueError("Class mapping names and indices are inconsistent.")


def build_class_mapping_from_values(values):
    class_names = sorted(values.dropna().unique().tolist())
    return {
        "class_names": class_names,
        "class_to_idx": {name: idx for idx, name in enumerate(class_names)},
        "num_classes": len(class_names),
    }

class ColonyDataset(Dataset):
    def __init__(self, csv_path, split, image_processor=None, transform=None, class_mapping=None):
        csv_path = Path(csv_path)
        self.df = _read_dataset_csv(csv_path)
        self.class_mapping = class_mapping or build_class_mapping_from_values(self.df["genus"])
        validate_class_mapping(self.df, self.class_mapping)
        _validate_image_files(self.df, csv_path)
        split_df = self.df[self.df["split"] == split].reset_index(drop=True)
        if split_df.empty:
            raise ValueError(f"Dataset split {split!r} in {csv_path} is empty.")
        missing_split_labels = set(self.class_mapping["class_names"]) - set(split_df["genus"])
        if missing_split_labels:
            raise ValueError(
                f"Dataset split {split!r} is missing labels: "
                f"{sorted(missing_split_labels)}."
            )
        self.genus_list = self.class_mapping["class_names"]
        self.genus_to_idx = self.class_mapping["class_to_idx"]

        self.df = split_df
        
        self.image_processor = image_processor
        self.transform = transform

    @property
    def num_classes(self):
        return len(self.genus_list)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = _resolve_image_path(row["filename"])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.image_processor is not None:
            encoded = self.image_processor(images=image, return_tensors="pt")
            image = encoded["pixel_values"].squeeze(0)

        label_index = self.genus_to_idx[row["genus"]]
        return image, torch.tensor(label_index, dtype=torch.long)

    def __len__(self):
        return len(self.df)



if __name__ == "__main__":
    image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
    dataset = ColonyDataset(
        csv_path=ROOT_DIR / "data" / "annotated" / "merged_labels_split.csv",
        split="train",
        image_processor=image_processor,
        transform=train_augment,
    )
    print("Dataset length:", len(dataset))
    image_tensor, label_index = dataset[0]
    print("Image shape:", image_tensor.shape)
    print("Label index:", label_index)
    print("Num classes:", dataset.num_classes)


