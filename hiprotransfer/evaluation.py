import numpy as np


def three_sigma_threshold(scores, sigma=3.0):
    """Return mean + sigma * std for a score vector."""
    scores = np.asarray(scores, dtype=np.float32)
    if scores.size == 0:
        raise ValueError("scores must not be empty")
    return np.float32(scores.mean() + float(sigma) * scores.std())


def predict_by_three_sigma(test_scores, reference_scores, sigma=3.0):
    """Predict anomalies in test_scores using a configurable sigma threshold."""
    threshold = three_sigma_threshold(reference_scores, sigma=sigma)
    pred = (np.asarray(test_scores, dtype=np.float32) > threshold).astype(np.int32)
    return pred, float(threshold)


def point_adjust(labels, predictions):
    """Apply point-adjustment over contiguous anomaly intervals."""
    labels = np.asarray(labels, dtype=np.int32)
    adjusted = np.asarray(predictions, dtype=np.int32).copy()
    if labels.shape != adjusted.shape:
        raise ValueError("labels and predictions must have the same shape")

    in_event = False
    start = 0
    for idx, value in enumerate(labels):
        if value == 1 and not in_event:
            start = idx
            in_event = True
        is_event_end = in_event and (value == 0 or idx == len(labels) - 1)
        if is_event_end:
            end = idx if value == 0 else idx + 1
            if adjusted[start:end].any():
                adjusted[start:end] = 1
            in_event = False
    return adjusted


def precision_recall_f1(labels, predictions):
    """Compute precision, recall, and F1 for binary anomaly labels."""
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.int32).reshape(-1)
    if labels.shape != predictions.shape:
        raise ValueError("labels and predictions must have the same shape")

    tp = int(((labels == 1) & (predictions == 1)).sum())
    fp = int(((labels == 0) & (predictions == 1)).sum())
    fn = int(((labels == 1) & (predictions == 0)).sum())

    precision = tp / float(tp + fp) if tp + fp > 0 else 0.0
    recall = tp / float(tp + fn) if tp + fn > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
