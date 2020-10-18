
from abc import ABCMeta,abstractmethod
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB

#Preprocesamiento de datos OneHot
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#Validaciones 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

import numpy as np
import pandas as pd

class Verificador:

    #Dbemos poder distinguir entre si es Gaussian o Multidimensional
    # Clase abstracta
    __metaclass__ = ABCMeta

    def preprocesado_Normal(self,filename): #LISTO A FALTA DE PRUEBA
        #Recuperamos CSV
        X = pd.read_csv(filename)

        #Mostramos para garantizar el archivo
        """print(X.shape)
        print(X.head(3))"""
        
        #Preprocesado de datos a etiquetas 
        le = LabelEncoder()
        X = X.apply(le.fit_transform)

        #Mostramos para garantizar el etiquetado
        """print(X.head(3))"""

        #Convertimos a array
        X_1 = X.to_numpy()

        #Mostramos con la nueva configuracion
        """print(X_1.shape)
        print(X_1)"""

        return X_1

    def preprocesado_OneHot(self,filename): #LISTO A FALTA DE PRUEBA
        #Recuperamos CSV
        X = pd.read_csv(filename)

        #Mostramos para garantizar el archivo
        """print(X.shape)
        print(X.head(3))"""

        #Preprocesado de datos a etiquetas 
        le = LabelEncoder()
        X = X.apply(le.fit_transform)

        #Mostramos para garantizar el etiquetado
        """print(X.head(3))"""

        #Declaramos preprocesado
        enc = OneHotEncoder(handle_unknown='ignore')

        #entrenamos encoder
        enc.fit(X)
        """print(enc.categories_)"""

        #Convertimos en tabla
        X_1 = enc.transform(X).toarray()

        #Mostramos con la nueva configuracion
        """print(X_1.shape)
        print(X_1)"""

        return X_1


    def separacion(self,datos):
        tam_fila = len(datos[0])-1
        tam = len(datos)
        clase = np.empty([tam])
        fila = np.empty([tam, tam_fila])
        for i in range(len(datos)):
            clase[i]=datos[i][tam_fila]
            fila[i] = np.delete(datos[i],tam_fila)
        return fila,clase

    def validacion_Simple(self,X,Y,porcentaje):
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=float(1)-porcentaje,shuffle=True)
        return X_train, X_test, Y_train, Y_test

    def validacion_Cruzada(self,clasificador,X,Y,folds):
        acierto_carpetas = cross_val_score(clasificador, X, Y, cv=folds)
        Y_pred = cross_val_predict(clasificador, X, Y, cv=folds)
        return acierto_carpetas, Y_pred



class Verificador_GaussianNB(Verificador):

    gnb = None
    datos = None
    pred = None

    def __init__(self,prepro,tipo_validacion,porcentaje,folds,archivo):
        self.gnb = GaussianNB()
        #Hacemos un preprocesado
        if(prepro == True):
            self.datos = self.preprocesado_OneHot(archivo)
        else: #mantenmos los datos sino
            self.datos = self.preprocesado_Normal(archivo)

        #EXTRACCION Y DE X
        X,Y=self.separacion(self.datos)

        #Tipo de Validacion
        if(tipo_validacion == 1):
            X_train, X_test, Y_train, Y_test = self.validacion_Simple(X,Y,porcentaje)
            #Entreamos el clasificador
            self.gnb.fit(X_train, Y_train)
            #Predecimos
            pred = self.gnb.predict(X_test)
            fallos = (Y_test != pred).sum()
            print("Verificador GaussianNB ==> Error medio Simple (%f) en archivo %s" % (float(fallos)/float(len(pred)),archivo))
        elif(tipo_validacion == 2):
            acierto_carpetas, pred = self.validacion_Cruzada(self.gnb,X,Y,folds)
            fallos = (Y != pred).sum()
            print("Verificador GaussianNB ==> Error medio Cruzado (%f) en archivo %s" % (float(fallos)/float(len(pred)),archivo))
            #print(acierto_carpetas)




class Verificador_Multinominal(Verificador):

    clf = None
    datos = None
    pred = None

    def __init__(self,prepro,tipo_validacion,porcentaje,folds,archivo,alpha=1.0,fit_prior=True):
        self.clf = MultinomialNB(alpha=alpha,fit_prior=fit_prior)
        if(prepro == True):
            self.datos = self.preprocesado_OneHot(archivo)
        else: #mantenmos los datos sino
            self.datos = self.preprocesado_Normal(archivo)

        #EXTRACCION Y DE X
        X,Y=self.separacion(self.datos)

        #Tipo de Validacion
        if(tipo_validacion == 1):
            X_train, X_test, Y_train, Y_test = self.validacion_Simple(X,Y,porcentaje)
            #Entreamos el clasificador
            self.clf.fit(X_train, Y_train)
            #Predecimos
            pred = self.clf.predict(X_test)
            fallos = (Y_test != pred).sum()
            print("Verificador MultinominalNB ==> Error medio Simple (%f) en archivo %s" % (float(fallos)/float(len(pred)),archivo))
        elif(tipo_validacion == 2):
            acierto_carpetas, pred = self.validacion_Cruzada(self.clf,X,Y,folds)
            fallos = (Y != pred).sum()
            print("Verificador MultinominalNB ==> Error medio Cruzado (%f) en archivo %s" % (float(fallos)/float(len(pred)),archivo))
            #print(acierto_carpetas)