import math
import numpy as np
from scipy.spatial.distance import mahalanobis


def Euclidea(datos, indice):
  num_filas = len(datos.datos)
  distancias = np.zeros((num_filas-1,2),dtype=float)
  num_atributos = len(datos.datos[0]) - 1
  aux = 0

  for i in range(num_filas):
    if(i != indice): # Mientras sean filas que no es la que estamos clasificando
      
      valor_total=0.0

      for j in range(num_atributos): # Sumamos las diferencias al cuadrado
        valor_total += math.pow((datos.datos[indice][j] - datos.datos[i][j]),2)

      # Hallamos la raiz del valor total   
      raiz = math.sqrt(valor_total)
      
      # Guardamos el valor de la distancia total y el indice al que corresponde el punto  
      distancias[aux] = [raiz, i]
      aux+=1
     
  # Teniendo ua todas las distancias, ordenamos de menor a mayor
  distancias = distancias[np.lexsort(np.fliplr(distancias).T)]

  return distancias

def Manhattan(datos, indice):
  num_filas = len(datos.datos)
  distancias = np.zeros((num_filas-1,2),dtype=float)
  num_atributos = len(datos.datos[0]) - 1
  aux = 0

  for i in range(num_filas):
    if(i != indice): # Mientras sean filas que no es la que estamos clasificando
      
      valor_total=0.0

      for j in range(num_atributos): # Sumamos las diferencias en valor absoluto
        valor_total += math.fabs(datos.datos[indice][j] - datos.datos[i][j])

      # Guardamos el valor de la distancia total y el indice al que corresponde el punto  
      distancias[aux] = [valor_total, i]
      aux+=1

  # Teniendo ua todas las distancias, ordenamos de menor a mayor
  distancias = distancias[np.lexsort(np.fliplr(distancias).T)]

  return distancias
  

def Mahalanobis(datos, indice):
  num_filas = len(datos.datos)
  distancias = np.zeros((num_filas-1,2),dtype=float)
  num_atributos = len(datos.datos[0]) - 1
  aux = 0

  atribs_indice = [float(atributo) for atributo in datos.datos[indice]]

  for i in range(num_filas):
    if(i != indice): # Mientras sean filas que no es la que estamos clasificando
      atribs_i = [float(atributo) for atributo in datos.datos[i]]

      # Calculamos la matriz de covarianza con la traspuesta de la matriz de las dos listas
      covarianza = np.cov(np.array([atribs_indice, atribs_i]).T)

      # Calculamos su inversa
      inversa = np.linalg.inv(covarianza)

      # Calculamos la distancia de mahalanobis
      valor_total = mahalanobis(atribs_indice, atribs_i, inversa)

      # Guardamos el valor de la distancia total y el indice al que corresponde el punto  
      distancias[aux] = [valor_total, i]
      aux+=1

  # Teniendo ua todas las distancias, ordenamos de menor a mayor
  distancias = distancias[np.lexsort(np.fliplr(distancias).T)]

  return distancias


def SeleccionKVecinos(datos, distancias, k):
  clases = []
  indice = 0
  
  # Para cada uno de los k vecinos que debemos seleccionar
  for i in range(k):
    # Sacamos el indice
    indice = int(distancias[i][1])
    # Guardamos la clasificacion del indice especificado
    clases.append(datos.datos[indice][-1])

    # Cuando nos encontramos en el ultimo vecino a saleccionar
    if i == (k-1):

      # Comprobamos que no haya empates en cuanto a distancia con los siguientes
      distancia_1 = distancias[i][0]
      j = i+1
      distancia_2 = distancias[j][0]

      # En caso de haberlos, los seleccionamos tambien como vecinos proximos
      while(distancia_2 == distancia_1):
        indice = int(distancias[i][1])
        clases.append(datos.datos[indice][-1]) 
        j+= 1 
        distancia_2 = distancias[j][0]

  # Devolvemos la clase que mas se repita
  return max(set(clases), key=clases.count)

