import numpy as np
from sklearn.metrics import precision_score, recall_score

DEFAULT_MIN_COVERAGE = 0.5


def _as_arrays(labels, predictions, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if not (len(labels) == len(predictions) == len(probabilities)):
        raise ValueError("Labels, predictions, and probabilities must have equal lengths.")
    if len(labels) == 0:
        raise ValueError("Threshold evaluation requires at least one sample.")
    return labels, predictions, probabilities


def threshold_tradeoff(
    labels,
    predictions,
    probabilities,
    threshold,
    num_classes,
):
    labels, predictions, probabilities = _as_arrays(
        labels, predictions, probabilities
    )
    accepted = probabilities >= float(threshold)
    accepted_count = int(accepted.sum())
    class_indices = list(range(num_classes))

    if accepted_count:
        precision = float(
            precision_score(
                labels[accepted],
                predictions[accepted],
                labels=class_indices,
                average="macro",
                zero_division=0,
            )
        )
    else:
        precision = 0.0

    rejected_predictions = np.where(accepted, predictions, -1)
    recall = float(
        recall_score(
            labels,
            rejected_predictions,
            labels=class_indices,
            average="macro",
            zero_division=0,
        )
    )
    total = len(labels)
    coverage = accepted_count / total
    return {
        "threshold": float(threshold),
        "macro_precision": precision,
        "macro_recall": recall,
        "coverage": coverage,
        "rejected": total - accepted_count,
        "rejection_rate": 1.0 - coverage,
        "accepted": accepted_count,
        "total": total,
    }


def select_global_threshold(
    labels,
    predictions,
    probabilities,
    num_classes,
    min_coverage=DEFAULT_MIN_COVERAGE,
):
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError("min_coverage must be greater than 0 and at most 1.")

    labels, predictions, probabilities = _as_arrays(
        labels, predictions, probabilities
    )
    candidates = np.unique(np.concatenate(([0.0], probabilities)))
    reports = [
        threshold_tradeoff(
            labels,
            predictions,
            probabilities,
            threshold,
            num_classes,
        )
        for threshold in candidates
    ]
    eligible = [report for report in reports if report["coverage"] >= min_coverage]
    if not eligible:
        raise ValueError("No threshold satisfies the requested minimum coverage.")

    selected = max(
        eligible,
        key=lambda report: (
            report["macro_precision"],
            report["macro_recall"],
            report["coverage"],
            -report["threshold"],
        ),
    )
    return {
        "method": "global_raw_softmax_predicted_probability",
        "source": "validation",
        "min_coverage": float(min_coverage),
        "selected": selected,
        "candidate_count": len(candidates),
    }


def apply_global_threshold(probabilities, threshold):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        raise ValueError("Probabilities must be a non-empty two-dimensional array.")
    predicted_indices = probabilities.argmax(axis=1)
    predicted_probabilities = probabilities.max(axis=1)
    uncertain = predicted_probabilities < float(threshold)
    return predicted_indices, predicted_probabilities, uncertain
