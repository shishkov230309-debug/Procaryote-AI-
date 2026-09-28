import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch
from PIL import Image
from torch import nn

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
import sys

sys.path.insert(0, str(SRC_DIR))

import app
import evaluate
import model
from checkpoint import validate_checkpoint_mapping
from dataset import ColonyDataset, build_class_mapping
from model import load_vit_checkpoint
from split_dataset import _validate_unique_images
from thresholding import apply_global_threshold, select_global_threshold


class FakeClassifierModel(nn.Module):
    def __init__(self, num_classes=19, layer_count=4):
        super().__init__()
        self.vit = nn.Module()
        self.vit.encoder = nn.Module()
        self.vit.encoder.layer = nn.ModuleList(
            [nn.Linear(4, 4) for _ in range(layer_count)]
        )
        self.classifier = nn.Linear(4, num_classes)
        self.config = SimpleNamespace(
            num_labels=num_classes,
            to_dict=lambda: {"model_type": "fake", "num_labels": num_classes},
        )

    def forward(self, pixel_values):
        features = torch.zeros(pixel_values.shape[0], 4)
        return SimpleNamespace(logits=self.classifier(features))


class ValidationTests(unittest.TestCase):
    def test_model_construction_freezing_and_forward_shape(self):
        fake_model = FakeClassifierModel()
        with patch.object(
            model.ViTForImageClassification,
            "from_pretrained",
            return_value=fake_model,
        ):
            classifier = model.create_vit_model(19, "fake-model")

        self.assertIs(classifier, fake_model)
        model.freeze_backbone(classifier, unfreeze_layers=2)
        for parameter in classifier.classifier.parameters():
            self.assertTrue(parameter.requires_grad)
        for layer in classifier.vit.encoder.layer[:2]:
            self.assertFalse(any(parameter.requires_grad for parameter in layer.parameters()))
        for layer in classifier.vit.encoder.layer[2:]:
            self.assertTrue(any(parameter.requires_grad for parameter in layer.parameters()))

        outputs = classifier(torch.zeros(3, 3, 224, 224))
        self.assertEqual(tuple(outputs.logits.shape), (3, 19))

    def test_strict_checkpoint_loading_and_architecture_mismatch(self):
        model_instance = nn.Linear(2, 2)
        metadata = {"class_names": ["a"], "num_classes": 1}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "checkpoint.pth"
            torch.save(
                {"state_dict": model_instance.state_dict(), "metadata": metadata},
                checkpoint_path,
            )
            loaded_metadata = load_vit_checkpoint(
                nn.Linear(2, 2), checkpoint_path, "cpu", expected_metadata=metadata
            )
            self.assertEqual(loaded_metadata, metadata)

            bad_path = Path(directory) / "bad.pth"
            torch.save(
                {"state_dict": nn.Linear(3, 2).state_dict(), "metadata": metadata},
                bad_path,
            )
            with self.assertRaises(RuntimeError):
                load_vit_checkpoint(
                    nn.Linear(2, 2), bad_path, "cpu", expected_metadata=metadata
                )

        with self.assertRaises(RuntimeError):
            evaluate._validate_model_architecture(
                SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {"width": 8})),
                {"model_config": {"width": 16}},
            )

    def test_class_mapping_consistency(self):
        mapping = {
            "class_names": ["a", "b"],
            "class_to_idx": {"a": 0, "b": 1},
            "num_classes": 2,
        }
        validate_checkpoint_mapping(mapping, mapping, "labels.csv")
        mismatched = dict(mapping, class_to_idx={"a": 1, "b": 0})
        with self.assertRaises(RuntimeError):
            validate_checkpoint_mapping(mismatched, mapping, "labels.csv")

    def _write_dataset_csv(self, directory, rows):
        path = Path(directory) / "labels.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["filename", "genus", "split"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def _write_image(self, directory, name):
        path = Path(directory) / name
        Image.new("RGB", (8, 8), color=(120, 80, 40)).save(path)
        return path

    def test_dataset_missing_file_empty_split_and_missing_class(self):
        with tempfile.TemporaryDirectory() as directory:
            image_a = self._write_image(directory, "a.png")
            image_b = self._write_image(directory, "b.png")
            image_c = self._write_image(directory, "c.png")
            missing_csv = self._write_dataset_csv(
                directory,
                [{"filename": str(Path(directory) / "missing.png"), "genus": "a", "split": "train"}],
            )
            with self.assertRaises(FileNotFoundError):
                ColonyDataset(missing_csv, "train")

            empty_csv = self._write_dataset_csv(
                directory,
                [{"filename": str(image_a), "genus": "a", "split": "train"}],
            )
            with self.assertRaises(ValueError):
                ColonyDataset(empty_csv, "val")

            missing_class_csv = self._write_dataset_csv(
                directory,
                [
                    {"filename": str(image_a), "genus": "a", "split": "train"},
                    {"filename": str(image_b), "genus": "b", "split": "train"},
                    {"filename": str(image_c), "genus": "a", "split": "val"},
                ],
            )
            mapping = build_class_mapping(missing_class_csv)
            with self.assertRaises(ValueError):
                ColonyDataset(missing_class_csv, "val", class_mapping=mapping)

    def test_split_leakage_detection(self):
        dataframe = pd.DataFrame(
            {
                "filename": ["train/sample.jpg", "test/SAMPLE.JPG"],
                "genus": ["a", "a"],
                "split": ["train", "test"],
            }
        )
        with self.assertRaises(ValueError):
            _validate_unique_images(dataframe)

    def test_metric_calculation_zero_prediction_class(self):
        metrics = evaluate.compute_metrics([0, 1, 1, 2], [0, 0, 0, 0], 3)
        self.assertEqual(metrics["support"], [1, 2, 1])
        self.assertEqual(metrics["prediction_counts"], [4, 0, 0])
        self.assertEqual(metrics["per_class_precision"], [0.25, 0.0, 0.0])
        self.assertAlmostEqual(metrics["macro_precision"], 0.25 / 3)

    def test_app_label_decoding_and_uncertain_behavior(self):
        class FakeProcessor:
            def __call__(self, images, return_tensors):
                return {"pixel_values": torch.zeros(1, 3, 2, 2)}

        class FakeModel:
            def __call__(self, **inputs):
                return SimpleNamespace(logits=torch.tensor([[0.0, 3.0]]))

        app.APPLICATION_STATE = app.ApplicationState(
            model=FakeModel(),
            image_processor=FakeProcessor(),
            device=torch.device("cpu"),
            class_names=["first_class", "second_class"],
            thresholding=None,
        )
        image = Image.new("RGB", (4, 4))
        prediction = app.predict_species(image)
        self.assertIn("Second Class", prediction)
        self.assertIn("predicted probability", prediction)

        app.APPLICATION_STATE.thresholding = {"selected": {"threshold": 0.99}}
        self.assertIn("Uncertain prediction", app.predict_species(image))
        app.APPLICATION_STATE = None

    def test_threshold_behavior_and_validation_provenance(self):
        labels = [0, 1, 1, 2]
        predictions = [0, 1, 2, 2]
        probabilities = [0.99, 0.90, 0.51, 0.20]
        calibration = select_global_threshold(
            labels, predictions, probabilities, 3, min_coverage=0.5
        )
        self.assertEqual(calibration["source"], "validation")
        report = calibration["selected"]
        self.assertGreaterEqual(report["coverage"], 0.5)
        _, _, uncertain = apply_global_threshold(
            [[0.9, 0.1], [0.4, 0.6]], report["threshold"]
        )
        self.assertEqual(len(uncertain), 2)


if __name__ == "__main__":
    unittest.main()
