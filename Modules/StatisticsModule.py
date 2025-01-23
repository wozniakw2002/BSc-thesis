from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve, classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from fpdf import FPDF
import seaborn as sns
class Statistics:
    
    @staticmethod
    def accuracy(Y: np.array, Y_pred: np.array) -> str:
        """
        This method counts accuracy value.

        Parameters:
        -----------
        Y: np.array -> array of true labels.
        Y_pred: np.array -> array of predicted labels.

        Returns:
        --------
        str -> metrice's value
        """

        acc = accuracy_score(Y,Y_pred) # (TP + TN)/(TP + TN + FP + FN)
        return ("%.2f"%round(acc,2)).replace('.', ',')
    
    @staticmethod
    def f1_score(Y: np.array, Y_pred: np.array) -> str:
        """
        This method counts f1-score.

        Parameters:
        -----------
        Y: np.array -> array of true labels.
        Y_pred: np.array -> array of predicted labels.

        Returns:
        --------
        str -> metrice's value
        """

        f1 = f1_score(Y, Y_pred) # 2TP/(2TP + FN + FP)
        return ("%.2f"%round(float(f1),2)).replace('.', ',')
    
    @staticmethod
    def preccision(Y: np.array, Y_pred: np.array) -> str:
        """
        This method counts preccision value.

        Parameters:
        -----------
        Y: np.array -> array of true labels.
        Y_pred: np.array -> array of predicted labels.

        Returns:
        --------
        str -> metrice's value
        """

        pre = precision_score(Y, Y_pred) # TP/(TP + FP)
        return ("%.2f"%round(float(pre),2)).replace('.', ',')
    
    @staticmethod
    def recall(Y: np.array, Y_pred: np.array) -> str:
        """
        This method counts recall value.

        Parameters:
        -----------
        Y: np.array -> array of true labels.
        Y_pred: np.array -> array of predicted labels.

        Returns:
        --------
        str -> metrice's value
        """

        rec = recall_score(Y, Y_pred) # TP/(TP + FN)
        return ("%.2f"%round(float(rec),2)).replace('.', ',')

    @staticmethod
    def auc(Y: np.array, Y_pred_prob: np.array) -> str:
        """
        This method counts AUC value.

        Parameters:
        -----------
        Y: np.array -> array of true labels.
        Y_pred_prob: np.array -> array of predicted probabilities.

        Returns:
        --------
        str -> metrice's value
        """

        fpr, tpr,_ = roc_curve(Y,Y_pred_prob)
        auc_val = auc(fpr, tpr)
        
        return ("%.2f"%round(float(auc_val),2)).replace(".", ".")

        
    @staticmethod
    def plot_roc_curve(Y: np.array, Y_pred_prob: np.array, save: bool = False, path:str = 'plots/roc', path2:str = '') -> None:
        """
        Plots the Receiver Operating Characteristic (ROC) curve for a model.

        Parameters:
        -----------
        Y: np.array
            Array of true binary labels (0 or 1).
        Y_pred_prob: np.array
            Array of predicted probabilities for the positive class.
        save: bool, optional
            Whether to save the plot to a file (default is False).
        path: str, optional
            File path to save the plot if `save` is True (default is 'plots/roc.png').

        Returns:
        --------
        None
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
            plt.savefig(path+path2+".png")
            return None
        else:
            plt.show()

    @staticmethod
    def plot_learning_curve(train_loss: np.array, val_loss: np.array, save: bool = False, path: str = 'plots/learn', path2:str = '') -> None:
        """
        Plots the learning curve, showing training and validation loss over epochs.

        Parameters:
        -----------
        train_loss: np.array
            Array of training loss values across epochs.
        val_loss: np.array
            Array of validation loss values across epochs.
        save: bool, optional
            Whether to save the plot to a file (default is False).
        path: str, optional
            File path to save the plot if `save` is True (default is 'plots/learn.png').

        Returns:
        --------
        None
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
            plt.savefig(path+path2+".png")
            return None
        else:
            plt.show()
    
    @staticmethod
    def plot_confusion_matrix(Y: np.array, Y_pred: np.array, save: bool = False, path: str = 'plots/conf', path2:str = '') -> None:
        """
        Plots the normalized confusion matrix for the predicted and true labels.

        Parameters:
        -----------
        Y: np.array
            Array of true binary labels (0 or 1).
        Y_pred: np.array
            Array of predicted binary labels (0 or 1).
        save: bool, optional
            Whether to save the plot to a file (default is False).
        path: str, optional
            File path to save the plot if `save` is True (default is 'plots/conf.png').

        Returns:
        --------
        None
        """

        conf_matrix = confusion_matrix(Y, Y_pred,normalize='all')
        fig, ax = plt.subplots(figsize=(5,5))
        sns.heatmap(conf_matrix,cmap="YlOrBr", annot=True, fmt='.2f', xticklabels=['Normal', 'Osteoarthritis'], yticklabels=['Normal', 'Osteoarthritis'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion matrix')
        if save:
            plt.savefig(path+path2+".png")
            return None
        else:
            plt.show()

    @staticmethod
    def plot_probability_histogram(Y_pred_prob: np.array, save: bool = False, path: str = 'plots/hist',path2:str = '') -> None:
        """
        Plots a histogram of predicted probabilities.

        Parameters:
        -----------
        Y_pred_prob: np.array
            Array of predicted probabilities for the positive class.
        save: bool, optional
            Whether to save the plot to a file (default is False).
        path: str, optional
            File path to save the plot if `save` is True (default is 'plots/hist.png').

        Returns:
        --------
        None
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
            plt.savefig(path+path2+".png")
            return None
        else:
            plt.show()
        

    @staticmethod
    def report(Y: np.array, Y_pred: np.array, Y_pred_prob: np.array, train_loss: np.array, val_loss: np.array, name: str) -> None:
        """
        Generates a comprehensive report for model evaluation, including plots and metrics.

        Parameters:
        -----------
        Y: np.array
            Array of true binary labels (0 or 1).
        Y_pred: np.array
            Array of predicted binary labels (0 or 1).
        Y_pred_prob: np.array
            Array of predicted probabilities for the positive class.
        train_loss: np.array
            Array of training loss values across epochs.
        val_loss: np.array
            Array of validation loss values across epochs.
        name: str
            Name used for saving the generated PDF report.

        Returns:
        --------
        None
        """
        Statistics.plot_confusion_matrix(Y, Y_pred, True, path2=name)
        Statistics.plot_learning_curve(train_loss, val_loss, True, path2=name)
        Statistics.plot_probability_histogram(Y_pred_prob, True, path2=name)
        Statistics.plot_roc_curve(Y, Y_pred_prob, True, path2=name)

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

        pdf.image('plots/hist'+name+".png", 10,45, 100)
        pdf.image('plots/roc'+name+".png", 110,45, 100)
        pdf.image('plots/conf'+name+".png", 10,145, 100)
        pdf.image('plots/learn'+name+".png", 110,145, 100)
        

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
            