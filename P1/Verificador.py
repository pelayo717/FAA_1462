
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

    #Debemos poder distinguir entre si es Gaussian o Multidimensional
    # Clase abstracta
    __metaclass__ = ABCMeta

    def preprocesado_Normal(self,filename):
        #Recuperamos CSV
        X = pd.read_csv(filename)

        #Preprocesado de datos a etiquetas 
        le = LabelEncoder()
        X = X.apply(le.fit_transform)

        #Convertimos a array
        X_1 = X.to_numpy()

        return X_1

    def preprocesado_OneHot(self,filename):
        #Recuperamos CSV
        X = pd.read_csv(filename)

        #Preprocesado de datos a etiquetas 
        le = LabelEncoder()
        X = X.apply(le.fit_transform)

        #Declaramos preprocesado
        enc = OneHotEncoder(handle_unknown='ignore')

        #Entrenamos encoder
        enc.fit(X)

        #Convertimos en tabla
        X_1 = enc.transform(X).toarray()

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
    real_pred = None


    def __init__(self):
        self.gnb = GaussianNB()

    def clasificate(self,prepro,tipo_validacion,porcentaje,folds,archivo):
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
            self.pred = self.gnb.predict(X_test)
            self.real_pred = Y_test
            fallos = (Y_test != self.pred).sum()
            return (float(fallos)/float(len(self.pred)))
        elif(tipo_validacion == 2):
            acierto_carpetas, self.pred = self.validacion_Cruzada(self.gnb,X,Y,folds)
            self.real_pred = Y
            fallos = (Y != self.pred).sum()
            return (float(fallos)/float(len(self.pred)))


class Verificador_Multinominal(Verificador):

    clf = None
    datos = None
    pred = None
    real_pred = None

    def __init__(self, alpha=1.0, fit_prior=True):
        self.clf = MultinomialNB(alpha=alpha,fit_prior=fit_prior)

    def clasificate(self, prepro, tipo_validacion, porcentaje, folds, archivo):
        if(prepro == True):
            self.datos = self.preprocesado_OneHot(archivo)
        else: #mantenmos los datos si no
            self.datos = self.preprocesado_Normal(archivo)

        #EXTRACCION Y DE X
        X,Y=self.separacion(self.datos)

        #Tipo de Validacion
        if(tipo_validacion == 1):
            X_train, X_test, Y_train, Y_test = self.validacion_Simple(X,Y,porcentaje)
            #Entreamos el clasificador
            self.clf.fit(X_train, Y_train)
            #Predecimos
            self.pred = self.clf.predict(X_test)
            self.real_pred = Y_test
            fallos = (Y_test != self.pred).sum()
            return float(fallos)/float(len(self.pred))
        elif(tipo_validacion == 2):
            acierto_carpetas, self.pred = self.validacion_Cruzada(self.clf,X,Y,folds)
            self.real_pred = Y
            fallos = (Y != self.pred).sum()
            return (float(fallos)/float(len(self.pred)))
