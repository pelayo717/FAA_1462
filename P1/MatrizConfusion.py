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

        
    
    def plot(self):
        print("EN DESARROLLO")
        """roc_auc = metrics.auc(self.fpr_array, self.tpr_array)
        plt.figure()
        plt.plot(self.fpr_array, self.tpr_array, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()"""
