"""Probability metrics shared by optional training and robustness evaluation."""
import math


def _summarize(items):
    if not items:
        raise ValueError("Metrics require at least one question")
    hits, tie_hits, kls, briers, confidences = [], [], [], [], []
    for item in items:
        prediction, gold = item["p"], item["g"]
        # Match the decoder: choose the first maximum if several options tie.
        predicted_index = max(range(len(prediction)), key=prediction.__getitem__)
        gold_index = max(range(len(gold)), key=gold.__getitem__)
        hits.append(int(predicted_index == gold_index))
        tie_hits.append(int(abs(gold[predicted_index] - max(gold)) < 1e-7))
        kls.append(sum(
            value * math.log(value / max(prediction[i], 1e-9))
            for i, value in enumerate(gold) if value > 0
        ))
        briers.append(sum((p - g) ** 2 for p, g in zip(prediction, gold)))
        confidences.append(max(prediction))

    count = len(items)
    ece = 0.0
    for bin_index in range(10):
        indices = [i for i, confidence in enumerate(confidences)
                   if min(int(confidence * 10), 9) == bin_index]
        if indices:
            accuracy = sum(hits[i] for i in indices) / len(indices)
            confidence = sum(confidences[i] for i in indices) / len(indices)
            ece += len(indices) / count * abs(accuracy - confidence)
    return {
        "decisions": count,
        "accuracy": sum(hits) / count,
        "tie_aware_accuracy": sum(tie_hits) / count,
        "kl_from_gold": sum(kls) / count,
        "brier": sum(briers) / count,
        "ece": ece,
    }


def metric_summary(items):
    """Report overall metrics and breakdowns by question type and workflow."""
    result = _summarize(items)
    for field in ("type", "workflow"):
        groups = {}
        for item in items:
            groups.setdefault(item[field], []).append(item)
        result["by_" + field] = {
            key: _summarize(groups[key]) for key in sorted(groups)
        }
    return result
