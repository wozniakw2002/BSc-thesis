from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from fpdf import FPDF
import seaborn as sns
class Statistics:
    
    @staticmethod
    def accuracy(Y: np.array, Y_pred: np.array) -> float:
        """
        This method counts accuracy value.

        Parameters:
        -----------
        Y: np.array -> array of true labels.
        Y_pred: np.array -> array of predicted labels.

        Returns:
        --------
        float -> metrice's value
        """

        acc = accuracy_score(Y,Y_pred) # (TP + TN)/(TP + TN + FP + FN)
        return ("%.2f"%round(acc,2)).replace('.', ',')
    
    @staticmethod
    def f1_score(Y: np.array, Y_pred: np.array) -> float:
        """
        This method counts f1-score.

        Parameters:
        -----------
        Y: np.array -> array of true labels.
        Y_pred: np.array -> array of predicted labels.

        Returns:
        --------
        float -> metrice's value
        """

        f1 = f1_score(Y, Y_pred) # 2TP/(2TP + FN + FP)
        return ("%.2f"%round(float(f1),2)).replace('.', ',')
    
    @staticmethod
    def preccision(Y: np.array, Y_pred: np.array) -> float:
        """
        This method counts preccision value.

        Parameters:
        -----------
        Y: np.array -> array of true labels.
        Y_pred: np.array -> array of predicted labels.

        Returns:
        --------
        float -> metrice's value
        """

        pre = precision_score(Y, Y_pred) # TP/(TP + FP)
        return ("%.2f"%round(float(pre),2)).replace('.', ',')
    
    @staticmethod
    def recall(Y: np.array, Y_pred: np.array) -> float:
        """
        This method counts recall value.

        Parameters:
        -----------
        Y: np.array -> array of true labels.
        Y_pred: np.array -> array of predicted labels.

        Returns:
        --------
        float -> metrice's value
        """

        rec = recall_score(Y, Y_pred) # TP/(TP + FN)
        return ("%.2f"%round(float(rec),2)).replace('.', ',')

    @staticmethod
    def auc(Y: np.array, Y_pred_prob: np.array) -> float:
        """
        This method counts AUC value.

        Parameters:
        -----------
        Y: np.array -> array of true labels.
        Y_pred_prob: np.array -> array of predicted probabilities.

        Returns:
        --------
        float -> metrice's value
        """

        fpr, tpr,_ = roc_curve(Y,Y_pred_prob)
        auc_val = auc(fpr, tpr)
        
        return "%.2f"%round(float(auc_val),2)

        
    @staticmethod
    def plot_roc_curve(Y: np.array, Y_pred_prob: np.array, save=False, path = 'plots/roc.png') -> None:
        """
        This method plots ROC curve of model.

        Parameters:
        -----------
        Y: np.array -> array of true labels.
        Y_pred_prob: np.array -> array of predicted probabilities.
        """

        plt.figure(figsize=(5,5))
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
        if save:
            plt.savefig(path)
            return None
        else:
            plt.show()

    @staticmethod
    def plot_learning_curve(train_loss: np.array, val_loss: np.array, save=False, path='plots/learn.png') -> None:
        """
        This method plots learning curve for training and validation set.

        Parameters:
        -----------
        trains_loss: np.array -> array of traning set loss values in consecutive epochs.
        val_loss: np.array -> array of validation set loss values in consecutive epochs.
        """

        plt.figure(figsize=(5,5))
        plt.plot(train_loss, label='train_loss')
        plt.plot(val_loss,label='val_loss')
        plt.title('Learning curve')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.xticks([i for i in range(1,len(train_loss) + 1, len(train_loss)//8)])
        plt.legend()
        if save:
            plt.savefig(path)
            return None
        else:
            plt.show()
    
    @staticmethod
    def plot_confusion_matrix(Y: np.array, Y_pred: np.array, save = False, path = 'plots/conf.png') -> None:
        """
        This method plots confusion matrix.

        Parameters:
        -----------
        Y: np.array -> array of true labels.
        Y_pred: np.array -> array of predicted labels.
        """

        conf_matrix = confusion_matrix(Y, Y_pred,normalize='all')
        fig, ax = plt.subplots(figsize=(5,5))
        sns.heatmap(conf_matrix,cmap="YlOrBr", annot=True, fmt='.2f', xticklabels=['Normall', 'Osteoarthritis'], yticklabels=['Normall', 'Osteoarthritis'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion matrix')
        if save:
            plt.savefig(path)
            return None
        else:
            plt.show()

    @staticmethod
    def plot_probability_histogram(Y_pred_prob: np.array, save = False, path='plots/hist.png') -> None:
        """
        This method plots histogram of probabilities.

        Parameters:
        -----------
        Y_pred_prob: np.array -> array of predicted probabilities.
        """

        plt.figure(figsize=(5,5))
        plt.hist(Y_pred_prob, 
                 bins=[i/10 for i in range(11)], 
                 weights=np.ones(len(Y_pred_prob)) / len(Y_pred_prob))
        plt.xlabel('Probability')
        plt.ylabel('Percent of predicted data')
        plt.title('Probability histogram')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.xticks([i/10 for i in range(11)])
        plt.xticks()
        if save:
            plt.savefig(path)
            return None
        else:
            plt.show()
        

    @staticmethod
    def report(Y, Y_pred, Y_pred_prob, train_loss, val_loss,name):
        Statistics.plot_confusion_matrix(Y, Y_pred, True)
        Statistics.plot_learning_curve(train_loss, val_loss, True)
        Statistics.plot_probability_histogram(Y_pred_prob, True)
        Statistics.plot_roc_curve(Y, Y_pred_prob, True)

        metrics = [Statistics.accuracy(Y, Y_pred), 
             Statistics.f1_score(Y,Y_pred),
             Statistics.preccision(Y,Y_pred),
             Statistics.recall(Y,Y_pred),
             Statistics.auc(Y,Y_pred_prob)]
        
        table = (
            ['Accuracy', 'F-score', 'Precision', 'Recall', 'AUC'],
            [str(el) for el in metrics]
        )

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Helvetica', 'b', 20)  
        pdf.cell(80)
        pdf.cell(20, 10, 'Model report', align='C')
        pdf.ln(30)

        pdf.set_font('Helvetica', 'b', 11)
        pdf.write(5, 'Vizualizations')

        pdf.image('plots/hist.png', 10,45, 100)
        pdf.image('plots/roc.png', 110,45, 100)
        pdf.image('plots/conf.png', 10,145, 100)
        pdf.image('plots/learn.png', 110,145, 100)
        

        pdf.ln(210)
        pdf.write(0, 'Statistics')
        pdf.ln(10)
        line_height = pdf.font_size * 2
        col_width = pdf.epw/5
        for row in table:
            for data in row:
                pdf.multi_cell(col_width, line_height, data, border=1, ln=3, max_line_height=3)
            pdf.ln(line_height)
        
        pdf.output(f"report_{name}.pdf", 'F')
            