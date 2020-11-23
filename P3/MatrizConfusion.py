import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LinearRegression


class MatrizConfusion():

    pred=None # Predicciones realizadas por el claisificador
    real_pred=None # Clasificacion correcta del dataset
    tpr=None # True Positives
    fnr=None # False Negatives
    fpr=None # False Positives
    tnr=None # True Negatives
    y=None # Array donde albergamos la clasificacion de cada entrada del dataset
    tpr_array=None 
    tpr_array=None
    
    def __init__(self):
        self.fpr_array=[]
        self.tpr_array=[]

    # Calcula la media de los aciertos y fallos del modelo y devuelve la tpr y la fpr
    def matrix_media(self, tp1, tp2, fp1, fp2, tn1, tn2, fn1, fn2):
        tp_media = (tp1 + tp2) / 2
        fp_media = (fp1 + fp2) / 2
        tn_media = (tn1 + tn2) / 2
        fn_media = (fn1 + fn2) / 2

        self.print_matrix(tp_media, fp_media, tn_media, fn_media)

        tpr = tp_media / (tp_media + fn_media)
        fpr = fp_media / (fp_media + tn_media)

        return tpr, fpr

    # Imprime la matriz de confusion
    def print_matrix(self, tp, fp, tn, fn):
        print("Matriz Confusion        Real")
        print("           |   1    " + str(round(tp, 3)) + "  " + str(round(fp, 3)))
        print(" Estimado  |   0    " + str(round(fn, 3)) + "  " + str(round(tn, 3)))

    def plot(self, point_list, dataset_name):
        
        # Colores de ploteo para los diferentes puntos
        plot_colors = ['g', 'r', 'c', 'm', 'y', 'k', 'b']

        plt.figure()
        # Dibujamos la diagonal
        plt.plot([0, 1], [0, 1], 'b--')
        # Limitamos la escala a [0, 1]
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])

        point_names = ["Normal"]

        # Ploteamos todos los puntos de la lista
        for i in range(len(point_list)):
            if i >= (len(plot_colors)):
                print("Demasiados puntos para plotear.")
                exit()
            # ftr, tpr
            plt.plot(point_list[i][0], point_list[i][1], plot_colors[i] + '+')
            point_names.append(point_list[i][2])

        # Aniadimos las etiquetas de los ejes
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        # Nombre de la grafica
        plt.title('Espacio ROC de ' + dataset_name)
        # Leyenda
        plt.legend(point_names, loc="lower right")
        plt.show()
