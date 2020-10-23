import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.linear_model import LinearRegression


class MatrizConfusion():

    pred=None
    real_pred=None
    tpr=None
    fnr=None
    fpr=None
    tnr=None
    y=None
    tpr_array=None
    tpr_array=None


    def __init__(self):
        self.fpr_array=[]
        self.tpr_array=[]

    def matrix_design(self,prediccion,real_prediccion):

        self.pred=prediccion
        self.real_pred=real_prediccion
        self.y=[]
        for o in range(len(self.real_pred)):
            self.y.append(self.real_pred[o][-1])

        #Contamos cuantos de cada clase
        valores_columna=[]
        for i in range(len(self.real_pred)):
            if(self.real_pred[i][-1] not in valores_columna):
                valores_columna.append(self.real_pred[i][-1])
        
        cnt_real = Counter()
        diccionario_fallos={}
        for j in range(len(valores_columna)): #Clase 1
            cnt_clase = Counter()
            for k in range(len(self.real_pred)): #Para todos los valores reales
                if(self.real_pred[k][-1]==valores_columna[j]): #este valor corresponde a la clase escogida en realidad
                    cnt_real[valores_columna[j]] +=1 
                    # cual es la prediccion hecha frente a esta entrada
                    if(self.real_pred[k][-1] == self.pred[k]):
                        cnt_clase[self.pred[k]] +=1
                    else:
                        cnt_clase[self.pred[k]] +=1
            diccionario_fallos[valores_columna[j]]=cnt_clase
        
        print("=> DATOS DE MATRIZ DE CONFUSION =>")
        print(valores_columna)
        print(cnt_real)
        print(diccionario_fallos)

        #PRESENTACION MATRIZ
        print("    " + str(valores_columna))
        for i in range(len(valores_columna)):
            fila=[]
            fila.append(valores_columna[i])
            aux = diccionario_fallos[valores_columna[i]]
            for j in range(len(valores_columna)):
                fila.append(aux[valores_columna[j]])
            print(fila)
        
        #hecho para dos clases solo
        aux = diccionario_fallos[valores_columna[0]]
        
        self.tpr = float(aux[valores_columna[0]]) / float(aux[valores_columna[0]] + aux[valores_columna[1]])
        self.fnr = float(aux[valores_columna[1]]) / float(aux[valores_columna[0]] + aux[valores_columna[1]])

        print(self.tpr) #eje Y
        print(self.fnr)

        #self.tpr_array.append(self.fnr)
        self.tpr_array.append(self.tpr)
        print(self.tpr_array)

        aux = diccionario_fallos[valores_columna[1]]

        self.fpr = float(aux[valores_columna[0]]) / float(aux[valores_columna[0]] + aux[valores_columna[1]])
        self.tnr = float(aux[valores_columna[1]]) / float(aux[valores_columna[0]] + aux[valores_columna[1]])

        print(self.fpr) #eje X
        print(self.tnr)
        #self.fpr_array.append(self.tnr)
        self.fpr_array.append(self.fpr)
        print(self.fpr_array)

    def matrix_media(self, tp1, tp2, fp1, fp2, tn1, tn2, fn1, fn2):
        tp_media = (tp1 + tp2) / 2
        fp_media = (fp1 + fp2) / 2
        tn_media = (tn1 + tn2) / 2
        fn_media = (fn1 + fn2) / 2

        self.print_matrix(tp_media, fp_media, tn_media, fn_media)

        tpr = tp_media / (tp_media + fn_media)
        fpr = fp_media / (fp_media + tn_media)

        return tpr, fpr

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
