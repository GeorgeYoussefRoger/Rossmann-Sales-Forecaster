import numpy as np

def rmspe(y_true, y_pred):
    """
    Root Mean Square Percentage Error
    """
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask])**2))