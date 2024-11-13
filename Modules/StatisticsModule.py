from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc
import numpy as np
class Statistics:
    
    def accuracy(Y: np.array, Y_pred: np.array) -> float:
        acc = accuracy_score(Y,Y_pred) # (TP + TN)/(TP + TN + FP + FN)
        return acc
    
    def f1_score(Y: np.array, Y_pred: np.array) -> float:
        f1 = f1_score(Y, Y_pred) # 2TP/(2TP + FN + FP)
        return float(f1)
    
    def preccision(Y: np.array, Y_pred: np.array) -> float:
        pre = precision_score(Y, Y_pred) # TP/(TP + FP)
        return float(pre)
    
    def recall(Y: np.array, Y_pred: np.array) -> float:
        rec = recall_score(Y, Y_pred) # TP/(TP + FN)
        return float(rec)

    # def auc(Y: np.array, Y_pred: np.array) -> float:
    #     auc_value = auc(Y, Y_pred) # TP/(TP + FN)
    #     return auc_value