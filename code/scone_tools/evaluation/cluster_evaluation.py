from sklearn.metrics import adjusted_rand_score
import numpy as np

def adjusted_rand_index(W_true, W_pred):
    true = np.argmax(W_true, axis=1)
    pred = np.argmax(W_pred, axis=1)
    return adjusted_rand_score(true, pred)

