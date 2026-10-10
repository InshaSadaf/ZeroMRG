"""Caption metrics with an explicit official/diagnostic boundary."""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence

from zeromrg.data.vocabulary import tokenize


def _ngrams(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def _corpus_bleu(references: Sequence[str], predictions: Sequence[str], order: int) -> float:
    clipped = [0] * order
    totals = [0] * order
    reference_length = 0
    prediction_length = 0
    for reference, prediction in zip(references, predictions):
        ref_tokens, pred_tokens = tokenize(reference), tokenize(prediction)
        reference_length += len(ref_tokens)
        prediction_length += len(pred_tokens)
        for n in range(1, order + 1):
            ref_counts, pred_counts = _ngrams(ref_tokens, n), _ngrams(pred_tokens, n)
            clipped[n - 1] += sum(min(count, ref_counts[gram]) for gram, count in pred_counts.items())
            totals[n - 1] += sum(pred_counts.values())
    if prediction_length == 0:
        return 0.0
    precisions = [(clipped[index] + 1.0) / (totals[index] + 1.0) for index in range(order)]
    brevity = 1.0 if prediction_length > reference_length else math.exp(1.0 - reference_length / prediction_length)
    return 100.0 * brevity * math.exp(sum(math.log(value) for value in precisions) / order)


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if left_token == right_token else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def diagnostic_caption_metrics(references: Sequence[str], predictions: Sequence[str]) -> dict[str, float | str]:
    """Dependency-free diagnostics; these are not paper-compatible COCO scores."""

    if not references or len(references) != len(predictions):
        raise ValueError("references and predictions must be non-empty and have equal length")
    result: dict[str, float | str] = {
        "metric_backend": "diagnostic_python_v1_not_paper_comparable",
    }
    for order in range(1, 5):
        result[f"BLEU-{order}"] = _corpus_bleu(references, predictions, order)
    rouge_scores = []
    exact = 0
    for reference, prediction in zip(references, predictions):
        ref_tokens, pred_tokens = tokenize(reference), tokenize(prediction)
        if reference == prediction:
            exact += 1
        common = _lcs_length(ref_tokens, pred_tokens)
        precision = common / len(pred_tokens) if pred_tokens else 0.0
        recall = common / len(ref_tokens) if ref_tokens else 0.0
        rouge_scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    result["ROUGE-L"] = 100.0 * sum(rouge_scores) / len(rouge_scores)
    result["exact_match_percent"] = 100.0 * exact / len(references)
    result["CIDEr"] = "unavailable_in_diagnostic_backend"
    result["METEOR"] = "unavailable_in_diagnostic_backend"
    return result


def official_caption_metrics(references: Sequence[str], predictions: Sequence[str]) -> dict[str, float | str]:
    """Compute the COCO-caption BLEU/CIDEr/ROUGE-L/METEOR metrics."""

    if not references or len(references) != len(predictions):
        raise ValueError("references and predictions must be non-empty and have equal length")
    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    except ImportError as exc:
        raise RuntimeError("Official evaluation requires pycocoevalcap and Java for METEOR") from exc
    refs = {index: [{"caption": text}] for index, text in enumerate(references)}
    preds = {index: [{"caption": text}] for index, text in enumerate(predictions)}
    refs, preds = PTBTokenizer().tokenize(refs), PTBTokenizer().tokenize(preds)
    bleu, _ = Bleu(4).compute_score(refs, preds)
    result: dict[str, float | str] = {"metric_backend": "pycocoevalcap_official"}
    result.update({f"BLEU-{index + 1}": 100.0 * float(value) for index, value in enumerate(bleu)})
    result["CIDEr"] = 100.0 * float(Cider().compute_score(refs, preds)[0])
    result["ROUGE-L"] = 100.0 * float(Rouge().compute_score(refs, preds)[0])
    result["METEOR"] = 100.0 * float(Meteor().compute_score(refs, preds)[0])
    return result
