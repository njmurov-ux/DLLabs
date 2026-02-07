import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    LPred = np.asarray(LPred)
    LTrue = np.asarray(LTrue)

    if LPred.shape != LTrue.shape:
        raise ValueError(f"LPred and LTrue must have the same shape, got {LPred.shape} vs {LTrue.shape}")

    if LTrue.size == 0:
        raise ValueError("Empty inputs: cannot compute accuracy.")

    acc = np.mean(LPred == LTrue)
    return float(acc)


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    LPred = np.asarray(LPred)
    LTrue = np.asarray(LTrue)

    if LPred.shape != LTrue.shape:
        raise ValueError(f"LPred and LTrue must have the same length, got {LPred.shape} vs {LTrue.shape}")

    labels = np.unique(np.concatenate([LPred, LTrue]))
    n = labels.size
    label_to_idx = {lab: i for i, lab in enumerate(labels)}

    cM = np.zeros((n, n), dtype=int)

    # rows = predicted, cols = true (actual)
    for p, t in zip(LPred, LTrue):
        cM[label_to_idx[p], label_to_idx[t]] += 1

    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    cM = np.asarray(cM)

    if cM.ndim != 2 or cM.shape[0] != cM.shape[1]:
        raise ValueError(f"cM must be a square 2D array, got shape {cM.shape}")

    total = cM.sum()
    if total == 0:
        raise ValueError("Confusion matrix has zero total count; cannot compute accuracy.")

    # correct predictions are on the diagonal; accuracy = trace / total
    acc = np.trace(cM) / total
    return float(acc)
