from abc import ABCMeta,abstractmethod
from EstrategiaParticionado import ValidacionCruzada,ValidacionSimple
import numpy as np
import pandas as pd
import math
import collections
import random
from Normalizar import *
from Distancias import *

class Clasificador:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
  def entrenamiento(self,datostrain, datostotales):
    pass
  
  
  @abstractmethod
  # devuelve un numpy array con las predicciones
  def clasifica(self, datostest, datostotales, tabla_clases, probabilidad_clase):
    pass
  
  
  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  def error(self, datostest, pred):
    tp = 0.0
    tn = 0.0
    fn = 0.0
    fp = 0.0
    aciertos = 0

    # La clase que tomaremos como "positiva" es la 1

    totales = len(datostest)
    for i in range(totales):
      if pred[i] == 1:
        if datostest[i][-1] == pred[i]:
          aciertos += 1
          tp += 1
        else:
          fp += 1
      else:
        if datostest[i][-1] == pred[i]:
          aciertos += 1
          tn += 1
        else:
          fn += 1
      
    tasa_aciertos = float(aciertos) / float(totales)
    
    return tasa_aciertos, tp, fp, tn, fn
    
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada
  def validacion(self, particionado, dataset, laplace=False, normalizacion_knn=True,seed=None):
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test. Otra opcion es repetir la 
    # validacion simple un numero especificado de veces, obteniendo en cada una un error. Finalmente se calculara la media.
    
    ########################################## Naive Bayes ##########################################

    if(isinstance(self, ClasificadorNaiveBayes) == True):
      datos_tabla_train = []
      datos_tabla_test = []
      lista_particiones = particionado.creaParticiones(dataset)
      media_error = 0.0
      media_tp = 0.0
      media_fp = 0.0
      media_tn = 0.0
      media_fn = 0.0
            
      # Recuperaremos tantas particiones como iteraciones se hayan indicado
      for i in range(len(lista_particiones)):
        
        # Creamos una tabla auxiliar con los datos de Train
        datos_tabla_train = dataset.extraeDatos(lista_particiones[i].indicesTrain,dataset.datos)
        
        # Creamos una tabla auxiliar con los datos de Test
        datos_tabla_test = dataset.extraeDatos(lista_particiones[i].indicesTest,dataset.datos)

        # LLamamos a la funcion de entrenamiento
        probabilidad_clase, analisis_atributos = self.entrenamiento(datos_tabla_train,dataset,laplace)

        # Llamamos a la funcion de clasificacion
        predicciones = self.clasifica(datos_tabla_test, dataset, analisis_atributos, probabilidad_clase)
        
        # Llamamos a la funcion de calculo del error y las tasas
        tasa_acierto, tp, fp, tn, fn = self.error(datos_tabla_test, predicciones)

        # Sumamos las tasas de fallo para calcular la media posteriormente
        media_error += (1 - tasa_acierto)

        # Sumamos las tasas tp, fp, tn y fn
        media_tp += tp
        media_fp += fp
        media_tn += tn
        media_fn += fn

    ########################################## KNN ##########################################


    if(isinstance(self, ClasificadorVecinosProximos) == True):
      datos_tabla_train = []
      datos_tabla_test = []      
      lista_particiones = particionado.creaParticiones(dataset)
      media_error = 0.0
      media_tp = 0.0
      media_fp = 0.0
      media_tn = 0.0
      media_fn = 0.0
      
      # LLamamos a la funcion de entrenamiento para normalizar todos los datos
      if(normalizacion_knn == True):
        datos_aux = self.entrenamiento(dataset)
      elif(normalizacion_knn == False):
        datos_aux = dataset.datos

      # Recuperaremos tantas particiones como iteraciones se hayan indicado
      for i in range(len(lista_particiones)):

        # Creamos una tabla auxiliar con los datos de Train
        datos_tabla_train = dataset.extraeDatos(lista_particiones[i].indicesTrain,datos_aux)
        
        # Creamos una tabla auxiliar con los datos de Test
        datos_tabla_test = dataset.extraeDatos(lista_particiones[i].indicesTest,datos_aux)
        
        # Llamamos a la funcion de clasificacion
        predicciones = self.clasifica(datos_tabla_train, datos_tabla_test)
                
        # Llamamos a la funcion de calculo del error y las tasas
        tasa_acierto, tp, fp, tn, fn = self.error(datos_tabla_test, predicciones)

        # Sumamos las tasas de fallo para calcular la media posteriormente
        media_error += (1 - tasa_acierto)

        # Sumamos las tasas tp, fp, tn y fn
        media_tp += tp
        media_fp += fp
        media_tn += tn
        media_fn += fn


    ######################################## Regresion Logistica ##########################################

    if(isinstance(self, ClasficadorRegresionLogistica) == True):
      datos_tabla_train = []
      datos_tabla_test = []
      lista_particiones = particionado.creaParticiones(dataset)
      media_error = 0.0
      media_tp = 0.0
      media_fp = 0.0
      media_tn = 0.0
      media_fn = 0.0
      
        # Recuperaremos tantas particiones como iteraciones se hayan indicado
      for i in range(len(lista_particiones)):
        
        # Creamos una tabla auxiliar con los datos de Train
        datos_tabla_train = dataset.extraeDatos(lista_particiones[i].indicesTrain,dataset.datos)
        
        # Creamos una tabla auxiliar con los datos de Test
        datos_tabla_test = dataset.extraeDatos(lista_particiones[i].indicesTest,dataset.datos)

        # LLamamos a la funcion de entrenamiento
        frontera_decision = self.entrenamiento(datos_tabla_train)

        # Llamamos a la funcion de clasificacion
        predicciones = self.clasifica(datos_tabla_test)
        
        # Llamamos a la funcion de calculo del error y las tasas
        tasa_acierto, tp, fp, tn, fn = self.error(datos_tabla_test, predicciones)
        
        # Sumamos las tasas de fallo para calcular la media posteriormente
        media_error += (1 - tasa_acierto)

        media_tp += tp
        media_fp += fp
        media_tn += tn
        media_fn += fn

    # Calculamos las medias 
    media_error = media_error / len(lista_particiones)
    media_tp =  media_tp / len(lista_particiones)
    media_fp =  media_fp / len(lista_particiones)
    media_tn =  media_tn / len(lista_particiones)
    media_fn =  media_fn / len(lista_particiones)

    return media_error, media_tp, media_fp, media_tn, media_fn

##############################################################################

class ClasificadorNaiveBayes(Clasificador):
  aux = None
  def __init__(self):
    self.aux = 0

  def entrenamiento(self,datostrain,datostotales,laplace=False):
    
    #Lista probabilidades a priori
    num_registros = len(datostrain)
    num_atributos = len(datostotales.nominalAtributos)-1

    #Para cada clase: Sacamos la probabilidad y variacion/varianza de cada atributo
    #con cada clase
    probabilidad_clase = {}

    #======= Primero Prob de clases ===========#
    for k in range (len(datostotales.diccionario["Class"])):
      #Valor de la clase recuperada
      valor_clase = list(datostotales.diccionario["Class"].items())[k][1]
      num_registros_clase = 0

      for l in range(num_registros):
        #Num ocurrencias de la clase concreta
        if(datostrain[l][num_atributos] == valor_clase):
          num_registros_clase+=1

      #Hallamos probabilidad P(Clase A) = num_registros_clase/num_registros
      prob=float(num_registros_clase/float(num_registros))
      probabilidad_clase[list(datostotales.diccionario["Class"].items())[k][0]]=prob

    #======= ANALIZAMOS CADA ATRIBUTO ===========#
    analisis_atributos={}
    for k in range (num_atributos):
      nombre_atributo=datostotales.atributos[k]

      #Atributo nominal o entero
      if(datostotales.nominalAtributos[k] == True): #Caso de Nominal
        #Matriz y conteo(debemos saber clases en eje Y y posibles valores en eje X)
        nombres_clases=[]
        valores_clases=[] #Eje Y
        for m in range (len(datostotales.diccionario["Class"])):
          #Valor de la clase recuperada
          nombres_clases.append(list(datostotales.diccionario["Class"].items())[m][0])
          valores_clases.append(list(datostotales.diccionario["Class"].items())[m][1])

        valores_posibles=[] #Eje X
        for n in range(len(datostotales.diccionario[nombre_atributo].items())):
          if(list(datostotales.diccionario[nombre_atributo].items())[n][1] not in valores_posibles):
            valores_posibles.append(list(datostotales.diccionario[nombre_atributo].items())[n][1])

        #Creamos matriz de dimensiones especificas
        matriz_atributo= np.empty((len(valores_clases), len(valores_posibles)))
        for l in range(len(valores_clases)):
          valores_columna=[]
          for o in range(num_registros):
              if(datostrain[o][num_atributos] == valores_clases[l]):
                valores_columna.append(datostrain[o][k])

          for p in range(len(valores_posibles)):
            matriz_atributo[l][p]=collections.Counter(valores_columna)[valores_posibles[p]]


        #Laplace
        if(laplace == True):
          if(np.count_nonzero(matriz_atributo) != len(valores_clases)*len(valores_posibles)): #No hay ceros
            matriz_atributo = matriz_atributo + 1
        
        analisis_atributos[nombre_atributo] = matriz_atributo
                
      else: #Caso de Entero/Real
        
        atributo={}
        for m in range(len(datostotales.diccionario["Class"])):
          nombre_clase = list(datostotales.diccionario["Class"].items())[m][0]
          valor_clase = list(datostotales.diccionario["Class"].items())[m][1]
          calculos={}
          media = 0
          varianza = 0
          lista_valores_clase = []
          for n in range(num_registros):
            if(datostrain[n][num_atributos] == valor_clase):
              lista_valores_clase.append(datostrain[n][k])

          # Hallamos media y varianza
          media = np.mean(lista_valores_clase)
          varianza = np.var(lista_valores_clase)
          calculos["media"]=media
          calculos["varianza"]=varianza
          atributo[nombre_clase]=calculos
        analisis_atributos[nombre_atributo] = atributo

    return (probabilidad_clase, analisis_atributos) 
     

  def clasifica(self, datostest, datostotales, analisis_atributos, probabilidad_clase):
    predicciones = []
    num_datos = len(datostest)                            # Numero de datos totales en test
    num_atributos = len(datostotales.nominalAtributos)-1  # Numero de atributos del dataset

    # Para cada uno de los datos de test
    for n in range(num_datos):
      probabilidades_post = {}
      
      # Para cada una de las hipotesis de la clase
      for k in range(len(datostotales.diccionario["Class"])):
        nombre_clase = list(datostotales.diccionario["Class"].items())[k][0]
        prob_priori = probabilidad_clase[nombre_clase]
        verosimilitudes = 1.0

        # Para cada uno de los atributos del dataset
        for x in range(num_atributos):

          ### Comprobamos si el atributo es nominal o numerico ###
          # Si el atributo es nominal
          valor_atributo = float(datostest[n][x])
          if datostotales.nominalAtributos[x] == True:
            
            tabla_atributo = analisis_atributos[datostotales.atributos[x]]  

            # Seleccionamos el valor del atributo
            for i in range(len(tabla_atributo[0])):
              # Cuando encontremos el valor del atributo, calculamos su verosimilitud

              if valor_atributo == float(list(datostotales.diccionario[datostotales.atributos[x]].items())[i][1]):
                
                # Calculamos el denominador 
                denominador = 0.0
                for j in range(len(tabla_atributo[k])):
                  denominador += tabla_atributo[k][j]

                verosimilitud_clase = tabla_atributo[k][i] / denominador
                break

          # Si el atributo es numerico
          else:
            varianza =  list(analisis_atributos[datostotales.atributos[x]].values())[k]['varianza']
            media =  list(analisis_atributos[datostotales.atributos[x]].values())[k]['media']

            # Control de errores en caso de que la varianza sea 0
            if varianza == 0.0:
              varianza = 0.000001 # Convertimos la varianza en 10^-6
            
            # Calculamos la verosimilitud de la clase
            verosimilitud_clase = 1 / (math.sqrt(2 * math.pi * varianza)) * math.exp( - pow(valor_atributo - media, 2) / (2*varianza))

          # Multiplicamos las probabilidades P(D=x|H=k)  
          verosimilitudes *= verosimilitud_clase 
        
        prob_posteriori = prob_priori * verosimilitudes

        probabilidades_post[list(datostotales.diccionario["Class"].items())[k][1]] = prob_posteriori

      predice = max(probabilidades_post, key=probabilidades_post.get)
      predicciones.append(predice)
    return predicciones


##############################################################################

class ClasificadorVecinosProximos(Clasificador):
  k = None
  distancia = None

  def __init__(self,k=3, distancia=Euclidea):
    self.k = k
    self.distancia = distancia

    if(self.distancia != Euclidea and self.distancia != Manhattan
                and self.distancia != Mahalanobis): 
      print("Calculo de distancia no permitido.")
      exit()
      

  def entrenamiento(self, datos):
    aux = Normalizar()
    aux.calcularMediasDesv(datos.datos,datos.nominalAtributos)
    datos_aux = aux.normalizarDatos(datos.datos,datos.nominalAtributos)
    return datos_aux
  
  def clasifica(self, datosTrain, datosTest):
    distancias = self.distancia(datosTrain, datosTest)
    predicciones = SeleccionKVecinos(datosTrain, distancias, self.k)
    return predicciones

##############################################################################

class ClasficadorRegresionLogistica(Clasificador):

  tasa_aprendizaje=None
  epocas=None
  frontera_decision=None

  # Recordemos que es recomendable que la tasa este entre -0.5 y 0.5
  def __init__(self,t_aprendizaje=0.2,epocas=10):
    self.tasa_aprendizaje = t_aprendizaje
    self.epocas = epocas
  
  def incializarFrontera(self):
    # Inicializamos los valores de la frontera aleatoriamente entre -0.5 y 0.5
    # Plantamos semilla
    random.seed(0)

    for i in range(len(self.frontera_decision)):
      self.frontera_decision[i] = random.uniform(-0.5,0.5)

  def entrenamiento(self,datos):
    # Necesitamos conocer el numero de atributos del dataset para saber cuantas componentes tendra la frontera
    atributos_vector_X = len(datos[0]) 
    self.frontera_decision = np.zeros(atributos_vector_X,dtype=float)
    self.incializarFrontera()

    for i in range(self.epocas):
      for j in range(len(datos)):
        # Calculamos el valor para introducir en la sigmoide
        # Multiplicacion de vectores <w*x>
        sumatorio = 0
        for k in range(atributos_vector_X): # Todos los atributos y Clase de la fila de datos
          if k == 0:
            sumatorio += 1.0 * self.frontera_decision[k]
          else:
            sumatorio += (float(datos[j][k-1]) * self.frontera_decision[k])

        # Pasamos por la sigmoide
        try:
          sigmoide = 1/(1+math.exp(-sumatorio))
        except OverflowError:
          if(sumatorio >= 0): #valor positivo
            sigmoide = 1.0 # e^(-sumatorio) siendo sumatorio mayor que 0 ==> un numero extremadamente pequenio ==> 1/1
          elif (sumatorio < 0):
            sigmoide = 0.0 # e^(-(-sumatorio)) siendo sumatorio menor que 0 ==> un numero extremadamente grande ==> 1/infinito ==> 0

        # Hacemos la correccion de frontera
        # Valor sigmoide - clase
        parte_auxiliar = (sigmoide - float(datos[j][-1]))*self.tasa_aprendizaje

        # Array axuiliar para guardar vector_X = tasa * (sigmoide-clasificacion) * vector_X
        vector_X_aux = np.zeros(atributos_vector_X,dtype=float)

        for k in range(atributos_vector_X):
          if k == 0:
            vector_X_aux[k] = 1.0 * parte_auxiliar
          else:
            vector_X_aux[k] = float(datos[j][k-1])*parte_auxiliar

          self.frontera_decision[k] = self.frontera_decision[k] - vector_X_aux[k]

    return self.frontera_decision
  
  def clasifica(self,datos):
    atributos_vector_X = len(datos[0])
    # Creamos un array para almacenar las predicciones
    predicciones = []
    # Recorremos todos las filas de datos
    for i in range(len(datos)):
      sumatorio = 0 # Sacamos el sumatorio de componentes del vector w por los atributos de la fila
      for k in range(atributos_vector_X): # Todos los atributos y Clase de la fila de datos
        if k == 0:
          sumatorio += 1.0 * self.frontera_decision[k]
        else:
          sumatorio += (float(datos[i][k-1]) * self.frontera_decision[k])

      # Pasamos por la sigmoide
      try:
        sigmoide = 1/(1+math.exp(-sumatorio))
      except OverflowError:
        if(sumatorio >= 0): #valor positivo
          sigmoide = 1.0 # e^(-sumatorio) siendo sumatorio mayor que 0 ==> un numero extremadamente pequenio ==> 1/1
        elif (sumatorio < 0):
          sigmoide = 0.0 # e^(-(-sumatorio)) siendo sumatorio menor que 0 ==> un numero extremadamente grande ==> 1/infinito ==> 0

      # Si P(X|C1) > 0.5
      if sigmoide >= 0.5: # La calculada en la sigmoide corresponde a la probabilidad de C1, entendemos que si esta es menor que 0,5 es por tanto menor que la C2 y por tanto el clasificador
                          # pensara que es de clase C2, y no de C1
        sigmoide = 1.0 # Clase 1
      else:
        sigmoide = 0.0 # Clase 2

      # Guardamos la prediccion
      predicciones.append(sigmoide)
    
    return predicciones



          



  