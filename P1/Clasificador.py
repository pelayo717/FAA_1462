#!/usr/bin/python

from abc import ABCMeta,abstractmethod
from EstrategiaParticionado import ValidacionCruzada,ValidacionSimple
import numpy as np
import pandas as pd
import math
import collections
#from tabulate import tabulate

class Clasificador:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
  def entrenamiento(self,datostrain, datostotales):
    pass
  
  
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto
  # devuelve un numpy array con las predicciones
  def clasifica(self, datostest, datostotales, tabla_clases, probabilidad_clase):
    pass
  
  
  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  def error(self, datostest, pred):
    aciertos = 0
    totales = len(datostest)
    for i in range(totales):
      if datostest[i][-1] == pred[i]:
        aciertos += 1
    tasa_aciertos = float(aciertos) / float(totales)
    
    return tasa_aciertos
    
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada

  """Entendemos que clasificador se refiere a Naive-Bayes o Knn (el nombre)"""

  def validacion(self,particionado,dataset,laplace=False,seed=None):
    # Creamos las particiones siguiendo la estrategia llamando a particionado.creaParticiones
    # - Para validacion cruzada: en el bucle hasta nv entrenamos el clasificador con la particion de train i
    # y obtenemos el error en la particion de test i
    # - Para validacion simple (hold-out): entrenamos el clasificador con la particion de train
    # y obtenemos el error en la particion test. Otra opcion es repetir la 
    # validacion simple un numero especificado de veces, obteniendo en cada una un error. Finalmente se calculara la media.
    if(isinstance(self, ClasificadorNaiveBayes) == True):
      datos_tabla_train = []
      datos_tabla_test = []
      lista_particiones = particionado.creaParticiones(dataset)
      media_error = 0.0
            
      # Recuperaremos tantas particiones como iteraciones se hayan indicado
      for i in range(len(lista_particiones)):
        
        # Creamos una tabla auxiliar con los datos de Train
        datos_tabla_train = dataset.extraeDatos(lista_particiones[i].indicesTrain)
        
        # Creamos una tabla auxiliar con los datos de Test
        datos_tabla_test = dataset.extraeDatos(lista_particiones[i].indicesTest)

        # LLamamos a la funcion de entrenamiento
        entrenamiento = self.entrenamiento(datos_tabla_train,dataset,laplace)
        probabilidad_clase = entrenamiento[0] # Probabilidad a Priori de las hipotesis
        analisis_atributos = entrenamiento[1] # Tablas de los atributos

        # Llamamos a la funcion de clasificacion
        predicciones = self.clasifica(datos_tabla_test, dataset, analisis_atributos, probabilidad_clase)
        
        # Llamamos a la funcion de calculo del error
        tasa_acierto = self.error(datos_tabla_test, predicciones)

        # Sumamos las tasas de fallo para calcular la media posteriormente
        media_error += (1 - tasa_acierto)

      media_error = media_error / len(lista_particiones)
      return media_error,datos_tabla_test,predicciones

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
        calculos={}
        atributo={}
        for m in range(len(datostotales.diccionario["Class"])):
          nombre_clase = list(datostotales.diccionario["Class"].items())[m][0]
          valor_clase = list(datostotales.diccionario["Class"].items())[m][1]
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
          if datostotales.nominalAtributos[x] == True:
            tabla_atributo = analisis_atributos[datostotales.atributos[x]]
            valor_atributo = float(datostest[n][x])

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
            verosimilitud_clase = 1 / (math.sqrt(2 * math.pi * varianza)) * math.exp( - pow(valor_atributo - media, 2) / 2*varianza)

          # Multiplicamos las probabilidades P(D=x|H=k)  
          verosimilitudes *= verosimilitud_clase 
        
        prob_posteriori = prob_priori * verosimilitudes

        probabilidades_post[list(datostotales.diccionario["Class"].items())[k][1]] = prob_posteriori

      predice = max(probabilidades_post, key=probabilidades_post.get)
      predicciones.append(predice)
    return predicciones