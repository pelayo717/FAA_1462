
from abc import ABCMeta,abstractmethod
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB,MultinomialNB

#P2 KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric

#P2 Regresion Logistica
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

#Preprocesamiento de datos OneHot
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#Preprocesamiento Regresion Logistica
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#Validaciones 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

import numpy as np
import pandas as pd

class Verificador:

    # Clase abstracta
    __metaclass__ = ABCMeta

    def preprocesado_Normal(self,filename):
        # Recuperamos CSV
        X = pd.read_csv(filename)

        # Preprocesado de datos a etiquetas 
        le = LabelEncoder()
        X = X.apply(le.fit_transform)

        # Convertimos a array
        X_1 = X.to_numpy()

        # Retornamos los datos, solamente etiquetados, como un preprocesado convencional
        return X_1

    def preprocesado_OneHot(self,filename):
        # Recuperamos CSV
        X = pd.read_csv(filename, dtype=str)

        # Preprocesado de datos a etiquetas 
        """le = LabelEncoder()
        X = X.apply(le.fit_transform)"""

        # Declaramos preprocesado One Hot
        enc = OneHotEncoder()

        # Entrenamos encoder
        enc.fit(X)

        # Convertimos en tabla
        X_1 = enc.transform(X).toarray()

        # Retornamos los datos, etiquetas con formato OneHot
        return X_1

    # Esta primitiva se encarga de separar de una fila de datos (tanto atributos como clasificacion),
    # de separar todos los datos de entrada del de salida, es decir de la clase asignada a esta entrada del dataset
    def separacion(self,datos):
        tam_fila = len(datos[0])-1
        tam = len(datos)
        clase = np.empty([tam])
        fila = np.empty([tam, tam_fila]) # Albergamos en distintos arrays, los atributos y la clase, y se retornan ambos
        for i in range(len(datos)):
            clase[i]=datos[i][tam_fila]
            fila[i] = np.delete(datos[i],tam_fila)
        return fila,clase

    # Esta funcion se encarga de llevar a cabo la validacion simple convencional, en la que hacemos uso de un porcentaje
    # concreto de entrenamiento, unos datos de entrada y unas clasificaciones
    # Permitimos siempre que los datos se asignen de manera aleatoria
    # Retorna, cuatro array con los atributos y la clasificacion, tanto de test como de train
    def validacion_Simple(self,X,Y,porcentaje):
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=float(1)-porcentaje,shuffle=True)
        return X_train, X_test, Y_train, Y_test

    # Esta funcion se encarga de realizar la validacion cruzada sobre el clasificador escogido y usando una serie de 
    # parametros de configuracion. Usamos el numero de carpetas escogido por el usuario, y necesitamos los datos de entrada
    # y clasificacion del dataset. 
    # Retorna las puntuaciones por carpeta y las predicciones realizadas
    def validacion_Cruzada(self,clasificador,X,Y,folds):
        acierto_carpetas = cross_val_score(clasificador, X, Y, cv=folds)
        Y_pred = cross_val_predict(clasificador, X, Y, cv=folds)
        return acierto_carpetas, Y_pred



class Verificador_GaussianNB(Verificador):

    gnb = None # Propio clasificador GaussianNB de ScikitLearn
    datos = None # Datos sin preprocesar
    pred = None # Predicciones realizadas
    real_pred = None # Clasificaciones reales


    def __init__(self):
        self.gnb = GaussianNB() # Iniciamos el clasificador

    def clasificate(self,prepro,tipo_validacion,porcentaje,folds,archivo):
        # Hacemos un preprocesado
        if(prepro == True):
            self.datos = self.preprocesado_OneHot(archivo)
        else: #mantenemos los datos sino
            self.datos = self.preprocesado_Normal(archivo)

        # EXTRACCION Y DE X
        X,Y=self.separacion(self.datos)

        # Tipo de Validacion
        if(tipo_validacion == 1):
            X_train, X_test, Y_train, Y_test = self.validacion_Simple(X,Y,porcentaje)
            # Entreamos el clasificador
            self.gnb.fit(X_train, Y_train)
            # Predecimos
            self.pred = self.gnb.predict(X_test)
            self.real_pred = Y_test
            # Hallamos el numero de fallos y retornamos
            fallos = (Y_test != self.pred).sum()
            return (float(fallos)/float(len(self.pred)))
        elif(tipo_validacion == 2):
            # Pasamos el clasificador y una serie de datos a la funcion de validacion cruzada
            acierto_carpetas, self.pred = self.validacion_Cruzada(self.gnb,X,Y,folds)
            self.real_pred = Y
            # Hallamos el numero de fallos y retornamos
            fallos = (Y != self.pred).sum()
            return (float(fallos)/float(len(self.pred)))


class Verificador_Multinominal(Verificador):

    clf = None # Propio clasificador MultinominalNB de Scikitlearn
    datos = None # Datos sin preprocesar
    pred = None # Predicciones realizadas
    real_pred = None # Clasificaciones reales

    # Asignamos la no equivalencia de probabilidad de clases (fit_prior) y la posibilidad de aplicar laplace con alpha
    def __init__(self, alpha=1.0, fit_prior=True): 
        self.clf = MultinomialNB(alpha=alpha,fit_prior=fit_prior) # Iniciamos el clasificador

    def clasificate(self, prepro, tipo_validacion, porcentaje, folds, archivo):
        # Hacemos un preprocesado
        if(prepro == True):
            self.datos = self.preprocesado_OneHot(archivo)
        else: # mantenmos los datos si no
            self.datos = self.preprocesado_Normal(archivo)

        # EXTRACCION Y DE X
        X,Y=self.separacion(self.datos)

        # Tipo de Validacion
        if(tipo_validacion == 1):
            X_train, X_test, Y_train, Y_test = self.validacion_Simple(X,Y,porcentaje)
            # Entreamos el clasificador
            self.clf.fit(X_train, Y_train)
            # Predecimos
            self.pred = self.clf.predict(X_test)
            self.real_pred = Y_test
            # Hallamos el numero de fallos y retornamos
            fallos = (Y_test != self.pred).sum()
            return float(fallos)/float(len(self.pred))
        elif(tipo_validacion == 2):
            # Pasamos el clasificador y una serie de datos a la funcion de validacion cruzada
            acierto_carpetas, self.pred = self.validacion_Cruzada(self.clf,X,Y,folds)
            self.real_pred = Y
            # Hallamos el numero de fallos y retornamos
            fallos = (Y != self.pred).sum()
            return (float(fallos)/float(len(self.pred)))

class Verificados_KVecinos(Verificador):

    kn = None
    datos = None # Datos sin preprocesar
    pred = None # Predicciones realizadas
    real_pred = None # Clasificaciones reales
    n_vecinos = None # Numero de vecinos a observar
    pesos = None 
    metrica = None # Tipo de distancia que se quiere calcular

    def __init__(self, nvecinos=3,pesos='uniform',metrica='euclidean'):
        self.n_vecinos = nvecinos
        self.pesos = pesos
        self.metrica = metrica
         

    def clasificate(self, prepro, tipo_validacion, porcentaje, folds, archivo):
        # Hacemos un preprocesado
        if(prepro == True):
            self.datos = self.preprocesado_OneHot(archivo)
        else: # mantenmos los datos si no
            self.datos = self.preprocesado_Normal(archivo)

        # EXTRACCION Y DE X
        X,Y=self.separacion(self.datos)

        if self.metrica == "mahalanobis":
            # Transformamos el dataset en un ndarray
            matrix =  np.zeros((len(self.datos),len(self.datos[0])-1),dtype=float)
            for i in range(len(self.datos)):
                aux = self.datos[i][:-1]
                matrix[i] = aux

            # Generamos la matriz de covarianza
            covarianza = np.cov(matrix.T)
  
            # Calculamos su inversa
            inversa = np.linalg.inv(covarianza)

            self.kn = KNeighborsClassifier(n_neighbors=self.n_vecinos,weights=self.pesos,metric=self.metrica, metric_params={'V': inversa})    
        
        else:
            self.kn = KNeighborsClassifier(n_neighbors=self.n_vecinos,weights=self.pesos,metric=self.metrica)

        # Tipo de Validacion
        if(tipo_validacion == 1):
            X_train, X_test, Y_train, Y_test = self.validacion_Simple(X,Y,porcentaje)
            # Entreamos el clasificador
            self.kn.fit(X_train, Y_train)
            # Predecimos
            self.pred = self.kn.predict(X_test)
            self.real_pred = Y_test
            # Hallamos el numero de fallos y retornamos
            fallos = (Y_test != self.pred).sum()
            return float(fallos)/float(len(self.pred))

        elif(tipo_validacion == 2):
            # Pasamos el clasificador y una serie de datos a la funcion de validacion cruzada
            acierto_carpetas, self.pred = self.validacion_Cruzada(self.kn,X,Y,folds)
            self.real_pred = Y
            # Hallamos el numero de fallos y retornamos
            fallos = (Y != self.pred).sum()
            return (float(fallos)/float(len(self.pred)))

# La diferencia es que LogisticRegression usa el metodo convencional para minimizar la funcion de coste, usando las 
# regularizaciones Lasso para ello. Por el contrario, SGDClassifier, es un clasififcador lineal generalizado,
# que usa el descenso del grandiente como resolucion. 
# Puede usar diferentes funciones de penalizacion, lo que permite configurar mucho mas el clasificador pero en si,
# ambos clasificadores resuelven lo mismo, pero de formas distintas.

class Verificados_RegresionLogistica_RL(Verificador):

    rl_rl = None # Propio clasificador LogisticRegression
    datos = None # Datos sin preprocesar
    pred = None # Predicciones realizadas
    pred_prob = None # Estimaciones de probabilidad
    real_pred = None # Clasificaciones reales

    # Recordamos que Lasso I1, usa para minimizar la funcion aniade una penalizacion C que corresponde a 
    # la media de la suma de los valores absolutos. Util para conocer relevancias entre atributos de entrada,
    # dicho de otro modo, para conocer cuales de los de entrada son irrelevantes.

    # Recodramod que Lasso I2, usa para minimizar la funcion de coste aniade una penalizacion C que corresponde
    # a la media por dos de la suma de valores absolutos al cuadrado. Util para conocer correlaciones entre
    # los atributos de entrada.

    # Solver es el algoritmo de optimizacion del clasififcador
    # ==> lbfgs: para conjuntos multiclase | L2 <== escogido predeterminado
    # ==> liblinear: para conjuntos de datos pequenios, uno vs resto 
    # ==> newton-cg: para conjuntos multiclase | L2
    # ==> sag,saga: para conjuntos de datos grandes y multiclase | L2 | L1 (saga)

    def __init__(self,penalizacion="l2",tolerancia=0.0001,constante=1.0,sesgo=True,iteraciones_maximas=100,metodo_resolucion = "lbfgs"):
        self.rl_rl = make_pipeline(StandardScaler(), LogisticRegression(penalty=penalizacion,tol=tolerancia,C=constante,fit_intercept=sesgo,max_iter=iteraciones_maximas,solver=metodo_resolucion))

    def clasificate(self,prepro,tipo_validacion,porcentaje,folds,archivo):
        # Hacemos un preprocesado
        if(prepro == True):
            self.datos = self.preprocesado_OneHot(archivo)
        else: # Mantenemos los datos sino
            self.datos = self.preprocesado_Normal(archivo)

        # EXTRACCION Y DE X
        X,Y=self.separacion(self.datos)

        # Tipo de Validacion
        if(tipo_validacion == 1):
            X_train, X_test, Y_train, Y_test = self.validacion_Simple(X,Y,porcentaje)
            # Entreamos el clasificador
            self.rl_rl.fit(X_train, Y_train)
            # Predecimos
            self.pred = self.rl_rl.predict(X_test)
            #self.pred_prob = self.rl_rl.predict_proba(X_test)
            self.real_pred = Y_test
            # Hallamos el numero de fallos y retornamos
            fallos = (Y_test != self.pred).sum()
            return (float(fallos)/float(len(self.pred)))
        elif(tipo_validacion == 2):
            # Pasamos el clasificador y una serie de datos a la funcion de validacion cruzada
            acierto_carpetas, self.pred = self.validacion_Cruzada(self.rl_rl,X,Y,folds)
            self.real_pred = Y
            # Hallamos el numero de fallos y retornamos
            fallos = (Y != self.pred).sum()
            return (float(fallos)/float(len(self.pred)))

class Verificados_RegresionLogistica_SGD(Verificador):

    rl_sgd = None # Propio clasificador LogisticRegression
    datos = None # Datos sin preprocesar
    pred = None # Predicciones realizadas
    real_pred = None # Clasificaciones reales

    # Recordamos que Lasso I1, usa para minimizar la funcion aniade una penalizacion C que corresponde a 
    # la media de la suma de los valores absolutos. Util para conocer relevancias entre atributos de entrada,
    # dicho de otro modo, para conocer cuales de los de entrada son irrelevantes.

    # Recodramod que Lasso I2, usa para minimizar la funcion de coste aniade una penalizacion C que corresponde
    # a la media por dos de la suma de valores absolutos al cuadrado. Util para conocer correlaciones entre
    # los atributos de entrada.

    # El parametro loss son los ditintos metodos que tiene este clasificador para clasificar, entre los que incluye
    # REGRESION LOGISTICA (log), REDES NEURONALES (perceptron) etc...

    def __init__(self,metodo="log",penalizacion="l2",alfa=0.0001,tolerancia=0.0001,sesgo=True,ratio_aprendizaje="optimal",iteraciones_maximas=100):
        self.rl_sgd = make_pipeline(StandardScaler(), SGDClassifier(loss=metodo,penalty=penalizacion,alpha=alfa,tol=tolerancia,fit_intercept=sesgo,learning_rate=ratio_aprendizaje,max_iter=iteraciones_maximas))

    def clasificate(self,prepro,tipo_validacion,porcentaje,folds,archivo):
        # Hacemos un preprocesado
        if(prepro == True):
            self.datos = self.preprocesado_OneHot(archivo)
        else: # Mantenemos los datos sino
            self.datos = self.preprocesado_Normal(archivo)

        # EXTRACCION Y DE X
        X,Y=self.separacion(self.datos)

        # Tipo de Validacion
        if(tipo_validacion == 1):
            X_train, X_test, Y_train, Y_test = self.validacion_Simple(X,Y,porcentaje)
            # Entreamos el clasificador
            self.rl_sgd.fit(X_train, Y_train)
            # Predecimos
            self.pred = self.rl_sgd.predict(X_test)
            self.real_pred = Y_test
            # Hallamos el numero de fallos y retornamos
            fallos = (Y_test != self.pred).sum()
            return (float(fallos)/float(len(self.pred)))
        elif(tipo_validacion == 2):
            # Pasamos el clasificador y una serie de datos a la funcion de validacion cruzada
            acierto_carpetas, self.pred = self.validacion_Cruzada(self.rl_sgd,X,Y,folds)
            self.real_pred = Y
            # Hallamos el numero de fallos y retornamos
            fallos = (Y != self.pred).sum()
            return (float(fallos)/float(len(self.pred)))


class Verificador_AlgoritmosGeneticos(Verificador):

    def __init__(self):
        return