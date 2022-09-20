import torch
from torchmetrics.functional import accuracy, precision, recall, f1_score
from data.shot import SHOT

from metrics.scorer import f1, macro, micro

metric_name_to_func = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1_score,
}

def get_true_prediction_with_certainty(y_hat, y, threshold=0.5, fallback=None):
    pred = (y_hat >= threshold).long()
    true_prediction, certainty = [], []
    for i, (p, t) in enumerate(zip(pred, y)):
        _pred = p.nonzero().squeeze(dim=-1).tolist()
        _cert = y_hat[i, _pred].tolist()
        if not _pred:
            if fallback is not None:
                _pred = [fallback]
                _cert = [1.0]
            else:
                _pred = [y_hat[i].argmax().item()]
                _cert = [y_hat[i].max().item()]
        _target = t.nonzero().squeeze(dim=-1).tolist()
        true_prediction.append((_target, _pred))
        certainty.append(_cert)
    return true_prediction, certainty

def get_true_prediction(y_hat, y, threshold=0.5, fallback=None):
    pred = (y_hat >= threshold).long()
    argmax = y_hat.argmax(dim=-1)
    true_prediction = []
    for i, (p, t) in enumerate(zip(pred, y)):
        _pred = p.nonzero().squeeze(dim=-1).tolist()
        if not _pred:
            _pred = [fallback] if fallback is not None else [argmax[i].item()]
        _target = t.nonzero().squeeze(dim=-1).tolist()
        true_prediction.append((_target, _pred))
    return true_prediction

def filter_true_prediction_by_labels(true_prediction, labels):
    filtered_true_prediction, num_predictions = [], 0
    for true_labels, predicted_labels in true_prediction:
        target = [l for l in true_labels if l in labels] 
        pred = [l for l in predicted_labels if l in labels]
        num_predictions += len(pred)
        filtered_true_prediction.append((target, pred))
    return filtered_true_prediction, num_predictions

def calc_metrics(y_hat, y, threshold=0.5, metrics=["choi", "shot", "global"], fallback=None):
    true_prediction = get_true_prediction(y_hat, y, threshold, fallback=fallback)
    
    # Choi metrics
    choi_metrics = dict()
    if "choi" in metrics:
        _, _, _avg_pred_n, macro_p, macro_r, macro_f1 = macro(true_prediction)
        _, _, _, micro_p, micro_r, micro_f1 = micro(true_prediction)

        choi_metrics = {
            "choi/avg-pred-n": _avg_pred_n,
            "choi/f1-micro": micro_f1,
            "choi/f1-macro": macro_f1,
            "choi/precision-macro": macro_p,
            "choi/precision-micro": micro_p,
            "choi/recall-macro": macro_r,
            "choi/recall-micro": micro_r,
        }

    # Shot metrics
    shot_metrics = dict()
    if "shot" in metrics:
        for shot, labels in SHOT.items():
            filtered_true_prediction, num_predictions = filter_true_prediction_by_labels(true_prediction, labels)
            _, _, _, macro_p, macro_r, macro_f1 = macro(filtered_true_prediction)
            _, _, _, micro_p, micro_r, micro_f1 = micro(filtered_true_prediction)
            _shot_metrics = {
                "f1-micro": micro_f1,
                "f1-macro": macro_f1,
                "precision-macro": macro_p,
                "precision-micro": micro_p,
                "recall-macro": macro_r,
                "recall-micro": micro_r,
                "num": float(num_predictions),
            }
            _shot_metrics = {f"shot-{shot}/{k}": v for k, v in _shot_metrics.items()}
            shot_metrics.update(_shot_metrics)
        if "shot-min_1/f1-macro" in shot_metrics and "shot-0/f1-macro" in shot_metrics:
            shot_metrics["shot-combined/f1-macro"] = f1(shot_metrics["shot-min_1/f1-macro"], shot_metrics["shot-0/f1-macro"])
            shot_metrics["shot-combined/f1-micro"] = f1(shot_metrics["shot-min_1/f1-micro"], shot_metrics["shot-0/f1-micro"])

    # Global metrics
    global_metrics = dict()
    if "global" in metrics:
        _y_hat = y_hat.detach().clone()
        if fallback is not None:
            s = (_y_hat >= threshold).sum(dim=1)
            for i in range(len(_y_hat)):
                if s[i] == 0:
                    _y_hat[i,fallback] = 1.0
        else:
            arg = _y_hat.argmax(dim=1)
            _y_hat[torch.arange(len(_y_hat)),arg] = 1

        num_classes = _y_hat.size(1)
        avg_pred_n = (_y_hat >= threshold).sum() / len(_y_hat)

        common_kwargs = dict(preds=_y_hat, target=y, num_classes=num_classes, threshold=threshold)
        global_metrics = {
            "avg-pred-n": avg_pred_n,
            "f1-micro": f1_score(average="micro", **common_kwargs).item(),
            "precision-micro": precision(average="micro", **common_kwargs).item(),
            "recall-micro": recall(average="micro", **common_kwargs).item(),
            "f1-macro": f1_score(average="macro", **common_kwargs).item(),
            "precision-macro": precision(average="macro", **common_kwargs).item(),
            "recall-macro": recall(average="macro", **common_kwargs).item(),
            "f1-samples": f1_score(average="samples", **common_kwargs).item(),
            "precision-samples": precision(average="samples", **common_kwargs).item(),
            "recall-samples": recall(average="samples", **common_kwargs).item(),
            "f1-weighted": f1_score(average="weighted", **common_kwargs).item(),
        }
    
    # Finalize
    result_metrics = dict(
        **global_metrics,
        **choi_metrics,
        **shot_metrics,
    )
    return result_metrics
