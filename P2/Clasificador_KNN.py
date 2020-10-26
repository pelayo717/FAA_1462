import math
import numpy as np


class Clasificador_KNN:
  
  k = None

  def __init__(self,k=3):
    self.k = k


  # PENSANDO QUE DEBEMOS DECIR EL INDICE COMO ARG
  def Ecuclidea(self,datos,indice):
    clasificar = datos.datos[indice]
    num_filas = len(datos.datos)
    distancias = np.zeros((num_filas-1,2),dtype=float)
    num_atributos = len(datos.datos[0]) - 1
    aux = 0
    resultados = np.zeros(len(datos.diccionario["Class"]),dtype=int)

    for i in range(num_filas):
      if(i != indice): #mientras sean filas que no es la que estamos clasificando
        
        array_info = np.zeros(2,dtype=float)
        valor_total=0

        for j in range(num_atributos): #hacemos las diferencias al cuadrado
          valor_total += math.pow((datos.datos[indice][j] - datos.datos[i][j]),2)
        #hallamos la raiz de aux y guardamos el valor    
        array_info[0] = math.sqrt(valor_total)
        #tb guardamos el indice al que corresponde
        array_info[1] = aux
        distancias[aux] = array_info
        aux+=1
        array_info = None
    
    # Comprobamos que se han insertado correctamente y ordenamos segun la primer componente de cada entrada en distancias
    # que se trata de la distancia hallada
    print(distancias)
    ind = np.lexsort((distancias[:,1],distancias[:,0]))
    print(distancias[ind])

    # RECOGER FRUTOS
    for i in range(self.k):
      indice = int(distancias[ind][i][1])
      for j in range(len(datos.diccionario["Class"])): #TENGO DUDAS
        if(datos.datos[indice][num_atributos] == datos.diccionario["Class"].items()[j][1]): # CORRESPONDE A LA CLASE
          print("Soy clase " + str(datos.diccionario["Class"].items()[j][1]))
          resultados[j] +=1
    print(resultados)
    for i in range(len(resultados)):
      resultados[i] =  float(resultados[i])/float(self.k)
        
    print(resultados)
    return resultados

  def Manhattan(self,datos,indice): #HAY QUE REPASAR
    clasificar = datos.datos[indice]
    num_filas = len(datos.datos)
    distancias = np.zeros((num_filas-1,2),dtype=float)
    num_atributos = len(datos.datos[0]) - 1
    aux = 0
    resultados = np.zeros(len(datos.diccionario["Class"]),dtype=int)

    for i in range(num_filas):
      if(i != indice): #mientras sean filas que no es la que estamos clasificando
        
        array_info = np.zeros(2,dtype=float)
        valor_total=0

        for j in range(num_atributos): #hacemos las diferencias al cuadrado
          valor_total += math.fabs(datos.datos[indice][j] - datos.datos[i][j])
        #hallamos la raiz de aux y guardamos el valor    
        array_info[0] = math.sqrt(valor_total)
        #tb guardamos el indice al que corresponde
        array_info[1] = aux
        distancias[aux] = array_info
        aux+=1
        array_info = None
    
    # Comprobamos que se han insertado correctamente y ordenamos segun la primer componente de cada entrada en distancias
    # que se trata de la distancia hallada
    print(distancias)
    ind = np.lexsort((distancias[:,1],distancias[:,0]))
    print(distancias[ind])

    # RECOGER FRUTOS
    for i in range(self.k):
      indice = int(distancias[ind][i][1])
      for j in range(len(datos.diccionario["Class"])): #TENGO DUDAS
        if(datos.datos[indice][num_atributos] == datos.diccionario["Class"].items()[j][1]): # CORRESPONDE A LA CLASE
          print("Soy clase " + str(datos.diccionario["Class"].items()[j][1]))
          resultados[j] +=1
    print(resultados)
    for i in range(len(resultados)):
      resultados[i] =  float(resultados[i])/float(self.k)
        
    print(resultados)
    return resultados

  def Mahalanobis(self):
    print("EN DESAROLLO")