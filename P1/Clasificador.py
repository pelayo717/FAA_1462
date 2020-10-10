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
  # TODO: implementar
  def error(self,datos,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error    
    pass
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada

  """Entendemos que clasificador se refiere a Naive-Bayes o Knn (el nombre)"""

  def validacion(self,particionado,dataset,seed=None):
    #Iterar X veces
      #Entrenar con conjunto de datos
      #Validar conjunto de datos
      #Obtener error

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
      if(isinstance(particionado,ValidacionSimple) == True):
        
          # Recuperaremos tantas particiones como iteraciones se hayan indicado
          for i in range(len(lista_particiones)):
            
            # Creamos una tabla auxiliar con los datos de Train
            num_registros = len(lista_particiones[i].indicesTrain)
            for j in range(num_registros):
              datos_tabla_train.append(dataset.datos[lista_particiones[i].indicesTrain[j]])
            
            # Creamos una tabla auxiliar con los datos de Test
            num_registros = len(lista_particiones[i].indicesTest)
            for j in range(num_registros):
              datos_tabla_test.append(dataset.datos[lista_particiones[i].indicesTest[j]])

            entrenamiento = self.entrenamiento(datos_tabla_train,dataset)
            probabilidad_clase = entrenamiento[0]
            tabla_clases = entrenamiento[1]

            self.clasifica(datos_tabla_test, dataset, tabla_clases, probabilidad_clase)
            
            

      elif(isinstance(particionado,ValidacionCruzada) == True):
          print("Por implementar")
      else:
          print("Error en el argumento particionado")  
    elif(self == "KNN"):
      print("En proceso")
    


    pass
##############################################################################

class ClasificadorNaiveBayes(Clasificador):
  aux = None
  def __init__(self):
    self.aux = 0

  # TODO: implementar
  def entrenamiento(self,datostrain,datostotales):
    
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
          nombres_clases.append(datostotales.diccionario["Class"].items()[m][0])
          valores_clases.append(datostotales.diccionario["Class"].items()[m][1])  

        valores_posibles=[] #Eje X
        for n in range(len(datostotales.diccionario[nombre_atributo].items())):
          if(datostotales.diccionario[nombre_atributo].items()[n][1] not in valores_posibles):
            valores_posibles.append(datostotales.diccionario[nombre_atributo].items()[n][1])

        #Creamos matriz de dimensiones especificas
        matriz_atributo= np.empty((len(valores_clases), len(valores_posibles)))
        for l in range(len(valores_clases)):
          valores_columna=[]
          for o in range(num_registros):
              if(datostrain[o][num_atributos] == valores_clases[l]):
                valores_columna.append(datostrain[o][k])


          for p in range(len(valores_posibles)):
            matriz_atributo[l][p]=collections.Counter(valores_columna)[valores_posibles[p]]

        if(np.count_nonzero(matriz_atributo) != len(valores_clases)*len(valores_posibles)): #No hay ceros
          matriz_atributo = matriz_atributo + 1
        
        analisis_atributos[nombre_atributo] = matriz_atributo


                
      else: #Caso de Entero/Real
        calculos={}
        atributo={}
        for m in range(len(datostotales.diccionario["Class"])):
          nombre_clase = datostotales.diccionario["Class"].items()[m][0]
          valor_clase = datostotales.diccionario["Class"].items()[m][1]
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

    print(analisis_atributos)
    return (probabilidad_clase, analisis_atributos) 
     
    
  # TODO: implementar
  def clasifica(self, datostest, datostotales, tabla_clases, probabilidad_clase):
    predicciones = []

    num_datos = len(datostest)
    num_atributos = len(datostotales.nominalAtributos)-1

    for l in range(num_datos):
      prob_posteriori = {}

      for i in range(len(datostotales.diccionario["Class"])):
        prob_clase = list(probabilidad_clase.items())[i][1]
        tabla_clase = list(tabla_clases.items())[i][1]

        verosimilitudes = 1.0

        for j in range(num_atributos):
          nombre_atributo = datostotales.atributos[j]
          valor_atributo = float(datostest[l][j])
          varianza =  tabla_clase[nombre_atributo]['varianza']
          media =  tabla_clase[nombre_atributo]['media']

          aux1=(math.sqrt(2 * math.pi * varianza) * math.exp(- ((valor_atributo - media)**2 / (2*varianza))))
          
          prob_atributo = 1 / aux1

          verosimilitudes *= prob_atributo
        
        prob_NB = prob_clase * verosimilitudes

        prob_posteriori[list(datostotales.diccionario["Class"].items())[i][1]] = prob_NB

      predice = max(prob_posteriori, key=prob_posteriori.get)
      predicciones.append(predice)
    print(predicciones)
    pass