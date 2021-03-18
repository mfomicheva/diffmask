import torch


def confusion_matrix(y_pred, y_true):
    device = y_pred.device
    labels = max(y_pred.max().item() + 1, y_true.max().item() + 1)

    return (
        (
            torch.stack((y_true, y_pred), -1).unsqueeze(-2).unsqueeze(-2)
            == torch.stack(
                (
                    torch.arange(labels, device=device).unsqueeze(-1).repeat(1, labels),
                    torch.arange(labels, device=device).unsqueeze(-2).repeat(labels, 1),
                ),
                -1,
            )
        )
        .all(-1)
        .sum(-3)
    )


def accuracy_precision_recall_f1(y_pred, y_true, average=True):
    M = confusion_matrix(y_pred, y_true)

    tp = M.diagonal(dim1=-2, dim2=-1).float()

    precision_den = M.sum(-2)
    precision = torch.where(
        precision_den == 0, torch.zeros_like(tp), tp / precision_den
    )

    recall_den = M.sum(-1)
    recall = torch.where(recall_den == 0, torch.ones_like(tp), tp / recall_den)

    f1_den = precision + recall
    f1 = torch.where(
        f1_den == 0, torch.zeros_like(tp), 2 * (precision * recall) / f1_den
    )

    return ((y_pred == y_true).float().mean(-1),) + (
        tuple(e.mean(-1) for e in (precision, recall, f1))
        if average
        else (precision, recall, f1)
    )


def matthews_corr_coef(y_pred, y_true):
    M = confusion_matrix(y_pred, y_true)
    assert sum(M.shape) == 4  # This is for binary classification only
    tn, fp, fn, tp = M.view(M.numel())
    numerator = (tp * tn) - (fp * fn)
    denominator = torch.sqrt(((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).to(torch.double))
    if denominator == 0:
        return 0
    else:
        return numerator / denominator
