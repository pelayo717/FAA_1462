import math
import numpy as np
from scipy.spatial.distance import mahalanobis


def Euclidea(datosTrain, datosTest):
  num_filas_train = len(datosTrain)
  num_filas_test = len(datosTest)
  distancias_totales = []
  num_atributos = len(datosTrain[0]) - 1

  # Para cada uno de los ejemplos de test
  for k in range(num_filas_test):
    distancias = np.zeros((num_filas_train,2),dtype=float)

    for i in range(num_filas_train):        
        valor_total=0.0

        for j in range(num_atributos): # Sumamos las diferencias al cuadrado
          valor_total += math.pow((datosTest[k][j] - datosTrain[i][j]),2)

        # Hallamos la raiz del valor total   
        raiz = math.sqrt(valor_total)
        
        # Guardamos el valor de la distancia total y el indice al que corresponde el punto  
        distancias[i] = [raiz, i]
    # Teniendo ua todas las distancias, ordenamos de menor a mayor
    distancias = distancias[np.lexsort(np.fliplr(distancias).T)]
    distancias_totales.append(distancias)
  return distancias_totales


def Manhattan(datosTrain, datosTest):
  num_filas_train = len(datosTrain)
  num_filas_test = len(datosTest)
  distancias_totales = []
  num_atributos = len(datosTrain[0]) - 1

  # Para cada uno de los ejemplos de test
  for k in range(num_filas_test):
    distancias = np.zeros((num_filas_train,2),dtype=float)

    for i in range(num_filas_train):        
      valor_total=0.0

      for j in range(num_atributos): # Sumamos las diferencias en valor absoluto
        valor_total += math.fabs(datosTest[k][j] - datosTrain[i][j])
        
      # Guardamos el valor de la distancia total y el indice al que corresponde el punto  
      distancias[i] = [valor_total, i]

    # Teniendo ua todas las distancias, ordenamos de menor a mayor
    distancias = distancias[np.lexsort(np.fliplr(distancias).T)]
    distancias_totales.append(distancias)
  return distancias_totales
 

def Mahalanobis(datosTrain, datosTest):
  num_filas_train = len(datosTrain)
  num_filas_test = len(datosTest)
  distancias_totales = []
  num_atributos = len(datosTrain[0]) - 1

  # Transformamos el dataset de Train a un ndarray
  matrix_train =  np.zeros((num_filas_train,num_atributos),dtype=float)
  for i in range(num_filas_train):
    aux = datosTrain[i][:-1]
    matrix_train[i] = aux

  # Generamos la matriz de covarianza
  covarianza = np.cov(matrix_train.T)
  
  # Calculamos su inversa
  inversa = np.linalg.inv(covarianza)

  # Para cada uno de los ejemplos de test
  for k in range(num_filas_test):
    distancias = np.zeros((num_filas_train,2),dtype=float)

    atribs_test = [float(atributo) for atributo in datosTest[k][:-1]]

    # Para cada uno de los ejemplos de train, calculamos su distancia al ejemplo de test
    for i in range(num_filas_train):
      atribs_train = [float(atributo) for atributo in datosTrain[i][:-1]]

      # Calculamos la distancia de mahalanobis
      valor_total = mahalanobis(atribs_test, atribs_train, inversa)

      # Guardamos el valor de la distancia total y el indice al que corresponde el punto  
      distancias[i] = [valor_total, i]

    # Teniendo ua todas las distancias, ordenamos de menor a mayor
    distancias = distancias[np.lexsort(np.fliplr(distancias).T)]
    distancias_totales.append(distancias)
  return distancias_totales


def SeleccionKVecinos(datosTrain, distancias, k):
  tamanioTest = len(distancias)
  predicciones = []
  
  # Para cada uno de los ejemplos de Test
  for x in range(tamanioTest):
    clases = []
    indice = 0
    # Para cada uno de los k vecinos que debemos seleccionar
    for i in range(k):
      # Sacamos el indice
      indice = int(distancias[x][i][1])
      # Guardamos la clasificacion del indice especificado
      clases.append(datosTrain[indice][-1])

      # Cuando nos encontramos en el ultimo vecino a saleccionar
      if i == (k-1):

        # Comprobamos que no haya empates en cuanto a distancia con los siguientes
        distancia_1 = distancias[x][i][0]
        j = i+1
        distancia_2 = distancias[x][j][0]

        # En caso de haberlos, los seleccionamos tambien como vecinos proximos
        while(distancia_2 == distancia_1):
          indice = int(distancias[x][i][1])
          clases.append(datosTrain[indice][-1]) 
          j+= 1 
          distancia_2 = distancias[x][j][0]

    # Predice la clase que mas se repita
    predice = max(set(clases), key=clases.count)
    predicciones.append(predice)

  return predicciones

