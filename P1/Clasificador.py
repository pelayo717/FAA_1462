#!/usr/bin/python

from abc import ABCMeta,abstractmethod
from EstrategiaParticionado import ValidacionCruzada,ValidacionSimple
import numpy as np
import pandas as pd
from tabulate import tabulate

class Clasificador:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Metodos abstractos que se implementan en casa clasificador concreto
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto
  # datosTrain: matriz numpy con los datos de entrenamiento
  # atributosDiscretos: array bool con la indicatriz de los atributos nominales
  # diccionario: array de diccionarios de la estructura Datos utilizados para la codificacion de variables discretas
  def entrenamiento(self,datosTrain,atributosDiscretos,diccionario):
    pass
  
  
  @abstractmethod
  # TODO: esta funcion debe ser implementada en cada clasificador concreto
  # devuelve un numpy array con las predicciones
  def clasifica(self,datosTest,atributosDiscretos,diccionario):
    pass
  
  
  # Obtiene el numero de aciertos y errores para calcular la tasa de fallo
  # TODO: implementar
  def error(self,datos,pred):
    # Aqui se compara la prediccion (pred) con las clases reales y se calcula el error    
	pass
    
    
  # Realiza una clasificacion utilizando una estrategia de particionado determinada

  """Entendemos que clasificador se refiere a Naive-Bayes o Knn (el nombre)"""

  # TODO: implementar esta funcion
  def validacion(self,particionado,dataset,clasificador,seed=None):
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
    if(clasificador == "Naive-Bayes"):
      lista_particiones = []
      datos_tabla = []
      lista_particiones = particionado.creaParticiones(dataset)
      if(isinstance(particionado,ValidacionSimple) == True):
          # Recuperaremos tantas particiones como iteraciones se hayan indicado
          for i in range(len(lista_particiones)):

            num_registros = len(lista_particiones[i].indicesTrain)
            num_atributos = len(dataset.atributos)-1
            #Para insertar todas las tuplas en una matriz
            #Para la particion concreta vemos todas las entradas (tuplas/indices de la tabla original) 
            for j in range(num_registros):
              datos_tabla.append(dataset.datos[lista_particiones[i].indicesTrain[j]])

            #Para cada clase: Sacamos la probabilidad y variacion/varianza de cada atributo
            #con cada clase
            probabilidad_clase = {}
            clase={}
            for k in range (len(dataset.diccionario["Class"])):
              #Entrada en el diccionario de la clase a estudiar
              clase[dataset.diccionario["Class"].items()[k][0]]={}

              #Valor de la clase recuperada
              valor_clase = dataset.diccionario["Class"].items()[k][1]
              num_registros_clase = 0

              for l in range(num_registros):
                #Num ocurrencias de la clase concreta
                if(datos_tabla[l][num_atributos] == valor_clase):
                  num_registros_clase+=1

              #Hallamos probabilidad P(Clase A) = num_registros_clase/num_registros
              prob=float(num_registros_clase/float(num_registros))
              probabilidad_clase[dataset.diccionario["Class"].items()[k][0]]=prob
            
              atributo={}
              #Para cada clase/atributo
              for l in range(num_atributos):
                nombre_atributo=dataset.atributos[l]                
                atributo[nombre_atributo]={}
                valores={}
                varianza=0
                media=0
                lista_valores_clase=[]
                #Sacamos valor de cada uno o sumamos todos los que encontremos en esta columna
                # Opcion B: Sumar valores
                #Primero la media
                for m in range(num_registros):
                  #Num ocurrencias de la clase concreta
                  if(datos_tabla[m][num_atributos] == valor_clase):
                    lista_valores_clase.append(datos_tabla[m][l])
                #hallamos media
                media = np.mean(lista_valores_clase)
                varianza = np.var(lista_valores_clase)
                valores["media"]=media
                valores["varianza"]=varianza
                atributo[nombre_atributo]=valores
              clase[dataset.diccionario["Class"].items()[k][0]]=atributo
            print(clase)
          





            """#Tendremos la tabla 
            tabla_aux=tabulate(datos_tabla,dataset.atributos,showindex=True)
            print(tabla_aux)"""
            

      elif(isinstance(particionado,ValidacionCruzada) == True):
          print("Por implementar")
      else:
          print("Error en el argumento particionado")  
    elif(clasificador == "KNN"):
      print("En proceso")
    


    pass
##############################################################################

class ClasificadorNaiveBayes(Clasificador):
  aux = None
  def __init__(self):
    self.aux = 0

  # TODO: implementar
  def entrenamiento(self,datostrain,atributosDiscretos,diccionario):
      #Lista probabilidades a priori
    pass
    
     
    
  # TODO: implementar
  def clasifica(self,datostest,atributosDiscretos,diccionario):
    pass