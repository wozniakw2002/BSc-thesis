from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
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

    @staticmethod
    def auc(Y: np.array, Y_pred_prob: np.array) -> float:
        """
        This method counts AUC value.

        Parameters:
        -----------
        Y: np.array -> array of true labels.
        Y_pred_prob: np.array -> array of predicted probabilities.
        """
        fpr, tpr,_ = roc_curve(Y,Y_pred_prob)
        auc_val = auc(fpr, tpr) # TP/(TP + FN)
        
        return float(auc_val)

    def report(Y, Y_pred):
        report = classification_report(Y, Y_pred)
        print(report)

        
    @staticmethod
    def plot_roc_curve(Y: np.array, Y_pred_prob: np.array) -> None:
        """
        This method plots ROC curve of model.

        Parameters:
        -----------
        Y: np.array -> array of true labels.
        Y_pred_prob: np.array -> array of predicted probabilities.
        """

        fpr, tpr,_ = roc_curve(Y,Y_pred_prob)
        auc_val = round(auc(fpr, tpr), 2)
        plt.plot(fpr, tpr, label = f'Out Model (AUC = {auc_val})')
        plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.title('ROC curve')
        plt.show()


    @staticmethod
    def plot_learning_curve(train_loss: np.array, val_loss: np.array) -> None:
        """
        This method plots learning curve for training and validation set.

        Parameters:
        -----------
        trains_loss: np.array -> array of traning set loss values in consecutive epochs.
        val_loss: np.array -> array of validation set loss values in consecutive epochs.
        """

        plt.plot(train_loss, label='train_loss')
        plt.plot(val_loss,label='val_loss')
        plt.title('Learning curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix(Y: np.array, Y_pred: np.array) -> None:
        """
        This method plots confusion matrix.

        Parameters:
        -----------
        Y: np.array -> array of true labels.
        Y_pred: np.array -> array of predicted labels.
        """

        matrix = confusion_matrix(Y, Y_pred)
        plot_matrix = ConfusionMatrixDisplay(confusion_matrix = matrix, display_labels = ['Healthy', 'Osteoarthritis'])
        plot_matrix.plot()
        plt.show()
    
    @staticmethod
    def plot_probability_histogram(Y_pred_prob: np.array) -> None:
        """
        This method plots histogram of probabilities.

        Parameters:
        -----------
        Y_pred_prob: np.array -> array of predicted probabilities.
        """
        plt.hist(Y_pred_prob, 
                 bins=[i/10 for i in range(11)], 
                 weights=np.ones(len(Y_pred_prob)) / len(Y_pred_prob))
        plt.xlabel('Probability')
        plt.ylabel('Percent of predicted data')
        plt.title('Probability histogram')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xticks([i/10 for i in range(11)])
        plt.xticks()
        plt.show()