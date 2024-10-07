
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score
import numpy as np

def compute_metrics(outputs, labels):
    precision = precision_score(np.argmax(labels, axis = 1), np.argmax(outputs, axis = 1))
    recall = recall_score(np.argmax(labels, axis = 1), np.argmax(outputs, axis = 1))
    f1 = f1_score(np.argmax(labels, axis = 1), np.argmax(outputs, axis = 1))   

    return precision, recall, f1 