from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve
import numpy as np
import matplotlib.pyplot as plt
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

    def roc_curve(Y, Y_pred):
        fpr, tpr, _ = roc_curve(Y, Y_pred) 
        roc_auc = auc(fpr, tpr)
        # Plot the ROC curve
        plt.figure()  
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Breast Cancer Classification')
        plt.legend()
        plt.show()
        