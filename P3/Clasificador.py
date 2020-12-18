from abc import ABCMeta,abstractmethod
from EstrategiaParticionado import ValidacionCruzada,ValidacionSimple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import collections
import random
from Normalizar import *
from Distancias import *
from Verificador import *

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
  def validacion(self, particionado, dataset, laplace=False, normalizacion_knn=True,seed=None, filename=None):
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

  ######################################## Algoritmo Genetico ##########################################

    if(isinstance(self, ClasficadorAlgoritmoGenetico) == True):
      datos_tabla_train = []
      datos_tabla_test = []
      lista_particiones = particionado.creaParticiones(dataset)
      media_error = 0.0
      media_tp = 0.0
      media_fp = 0.0
      media_tn = 0.0
      media_fn = 0.0

      # Creamos instancia de Verificador Para realizar el preprocesado One Hot
      verificador = Verificador_AlgoritmosGeneticos()

      # Codificamos los datos con One Hot
      x = verificador.preprocesado_OneHot(filename)
      
      # Creamos una tabla auxiliar con los datos de Train
      datos_tabla_train = dataset.extraeDatos(lista_particiones[0].indicesTrain,x)
      
      # Creamos una tabla auxiliar con los datos de Test
      datos_tabla_test = dataset.extraeDatos(lista_particiones[0].indicesTest,x)  
      
      mejor_individuo, graf_media, graf_mejor = self.entrenamiento(dataset, datos_tabla_train)

      print("Mejor individuo: " + str(mejor_individuo))

      # Ploteamos las graficas de fitness
      self.plotear_graficas(graf_media, graf_mejor, filename)

      predicciones = self.clasifica(mejor_individuo, datos_tabla_test)

      # Llamamos a la funcion de calculo del error y las tasas
      tasa_acierto, tp, fp, tn, fn = self.error(datos_tabla_test, predicciones)

      # Sumamos las tasas de fallo para calcular la media posteriormente
      media_error += (1 - tasa_acierto)

      return media_error, tp, fp, tn, fn


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

      # Atributo nominal o entero
      if(datostotales.nominalAtributos[k] == True): #Caso de Nominal
        # Matriz y conteo(debemos saber clases en eje Y y posibles valores en eje X)
        nombres_clases=[]
        valores_clases=[] #Eje Y
        for m in range (len(datostotales.diccionario["Class"])):
          # Valor de la clase recuperada
          nombres_clases.append(list(datostotales.diccionario["Class"].items())[m][0])
          valores_clases.append(list(datostotales.diccionario["Class"].items())[m][1])

        valores_posibles=[] #Eje X
        for n in range(len(datostotales.diccionario[nombre_atributo].items())):
          if(list(datostotales.diccionario[nombre_atributo].items())[n][1] not in valores_posibles):
            valores_posibles.append(list(datostotales.diccionario[nombre_atributo].items())[n][1])

        # Creamos matriz de dimensiones especificas
        matriz_atributo= np.empty((len(valores_clases), len(valores_posibles)))
        for l in range(len(valores_clases)):
          valores_columna=[]
          for o in range(num_registros):
              if(datostrain[o][num_atributos] == valores_clases[l]):
                valores_columna.append(datostrain[o][k])

          for p in range(len(valores_posibles)):
            matriz_atributo[l][p]=collections.Counter(valores_columna)[valores_posibles[p]]

        # Laplace
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

  # Inicializamos la clase de vecinos proximos pasando la funcion de distancia escogida
  # y el numero de vecinos a tener en cuenta
  def __init__(self,k=3, distancia=Euclidea):
    self.k = k
    self.distancia = distancia

    # Verificamos que la funcion de distancia escogida sea una de las permitidas
    if(self.distancia != Euclidea and self.distancia != Manhattan
                and self.distancia != Mahalanobis): 
      print("Calculo de distancia no permitido.")
      exit()
      
  # Durante el entrenamiento nos encargamos de normalizar aquellos atributos que sean de tipo decimal
  # Para ello hacemos uso del moodulo Normalizar.py
  def entrenamiento(self, datos):
    aux = Normalizar()
    aux.calcularMediasDesv(datos.datos,datos.nominalAtributos)
    datos_aux = aux.normalizarDatos(datos.datos,datos.nominalAtributos)
    return datos_aux
  
  # Usamos el modulo externo de Distancias.py para llevar a cabo los calculos sobre las funciones
  # de distancia y para mas tarde recuperar las predicciones sobre las instancias escogidas
  def clasifica(self, datosTrain, datosTest):
    distancias = self.distancia(datosTrain, datosTest)
    predicciones = SeleccionKVecinos(datosTrain, distancias, self.k)
    return predicciones

##############################################################################

class ClasficadorRegresionLogistica(Clasificador):

  tasa_aprendizaje=None
  epocas=None
  frontera_decision=None
  lista_decisiones=None

  # Recordemos que es recomendable que la tasa este entre -0.5 y 0.5
  # Inicializamos la clase pasando el numero de epocas, que de manera predeterminada esta a 10,
  # y la tasa de aprendizaje que se encuentra en el 0.2 de serie
  # Ademas inicializamos el array de decisiones de posibles valores
  def __init__(self,t_aprendizaje=0.2,epocas=10):
    self.tasa_aprendizaje = t_aprendizaje
    self.epocas = epocas
    self.lista_decisiones=[]
  
  def incializarFrontera(self):
    # Inicializamos los valores de la frontera aleatoriamente entre -0.5 y 0.5
    # Plantamos semilla
    random.seed()
    for i in range(len(self.frontera_decision)):
      self.frontera_decision[i] = random.uniform(-0.5,0.5)

  def entrenamiento(self,datos):
    # Necesitamos conocer el numero de atributos del dataset para saber cuantas componentes tendra la frontera
    atributos_vector_X = len(datos[0]) 
    self.frontera_decision = np.zeros(atributos_vector_X,dtype=float)
    self.incializarFrontera()

    # Guardamos las clases para luego despues de realizar la sigmoidal asignar en el cambio de peso la clase correspondiente
    for i in range(len(datos)):
      if datos[i][atributos_vector_X-1] not in self.lista_decisiones:
        self.lista_decisiones.append(datos[i][atributos_vector_X-1])

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
          if(sumatorio > 0): #valor positivo
            sigmoide = 1.0 # e^(-sumatorio) siendo sumatorio mayor que 0 ==> un numero extremadamente pequenio ==> 1/1
          elif (sumatorio < 0):
            sigmoide = 0.0 # e^(-(-sumatorio)) siendo sumatorio menor que 0 ==> un numero extremadamente grande ==> 1/infinito ==> 0

        # Hacemos la correccion de frontera
        # Valor sigmoide - clase
        if(datos[j][-1] == self.lista_decisiones[0]):
          t=1.0
        else:
          t=0.0

        parte_auxiliar = (sigmoide - t)*self.tasa_aprendizaje

        # Array axuiliar para guardar vector_X = tasa * (sigmoide-clasificacion) * vector_X
        vector_X_aux = np.zeros(atributos_vector_X,dtype=float)

        for k in range(atributos_vector_X):
          if k == 0:
            vector_X_aux[k] = 1.0 * parte_auxiliar
          else:
            vector_X_aux[k] = float(datos[j][k-1])*parte_auxiliar

          # Generamos la nueva frontera de decision asignando los nuevos valores a las componentes correspondientes
          self.frontera_decision[k] = self.frontera_decision[k] - vector_X_aux[k]

    # Devolvemos la frontera de decision
    return self.frontera_decision
  
  def clasifica(self,datos):
    atributos_vector_X = len(datos[0])
    
    # Creamos un array para almacenar las predicciones
    predicciones = []
    
    # Recorremos todas las filas de datos
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
      if sigmoide > 0.5: # La calculada en la sigmoide corresponde a la probabilidad de C1, entendemos que si esta es menor que 0'5, entonces es menor que la C2 y por tanto el clasificador
                          # pensara que es de clase C2, y no de C1
        sigmoide = self.lista_decisiones[0]
      else:
        sigmoide =self.lista_decisiones[1]

      # Guardamos la prediccion
      predicciones.append(sigmoide)
    
    # Devolvemos las predicciones realizadas
    return predicciones



##############################################################################

class ClasficadorAlgoritmoGenetico(Clasificador):

  tamanio_poblacion=None          # Numero de individuos de la poblacion
  condicion_terminacion=None      # Numero de etapas maxima para la finalizacion
  maximo_reglas_individuo=None    # Numero maximo de reglas que puede tener un individuo
  elitismo = None                 # % de elitismo para pasar en la Seleccion de Progenitores
  poblacion = []                  # Poblacion del algoritmo
  valoresXAtributo = []           # Array con la cantidad de valores que pueden tomar los atributos del dataset
  longuitud_regla = 0             # Tamanio de las reglas de los individuos
  cruce = None                    # Tipo de funcion de cruce
  mutacion = None                 # Tipo de funcion de mutacion
  prob_cruce = 0.25               # Probabilidad de cruce del clasificador
  prob_mutacion = 0.05            # Probabilidad de mutacion del clasificador
  
  # Inicializamos el clasificador, pasandole como argumentos el tamanio de la poblacion, 
  # la condicion de finalizacion (num de epocas) y el maximo numero de reglas que puede
  # tener un individuo.

  def __init__(self, tam_poblacion=100, cond_terminacion=150, max_reglas=25, 
              tipo_cruce=0, tipo_mutacion=0, prob_cruce=0.25, prob_mutacion=0.05):
    self.tamanio_poblacion = tam_poblacion
    self.condicion_terminacion = cond_terminacion
    self.maximo_reglas_individuo = max_reglas
    self.prob_cruce = prob_cruce
    self.prob_mutacion = prob_mutacion

    if tipo_cruce == 0:
      self.cruce = self.cruce_intra
    else:
      self.cruce = self.cruce_inter

    if tipo_mutacion == 0:
      self.mutacion = self.mutacion_bitflip
    elif tipo_mutacion == 1:
      self.mutacion = self.mutacion_add_regla
    else:
      self.mutacion = self.mutacion_remove_regla

    self.elitismo = 5 # Por defecto, elegimos un 5% de elitismo

  # Generacion de la poblacion inicial
  def inizializarPoblacion(self, datos):
    numAtributos = len(datos.atributos) # Cantidad de atributos del dataset
    valoresAtributos = []               # Array con la cantidad de posibles valores que toman los 
                                        # atributos segun su posicion

    random.seed() # Inicializamos la semilla, necesaria para el uso de rand

    # Se da por hecho que todos los atributos seran de tipo nominal
    for atributo in datos.atributos[:-1]:
      tam = len(datos.diccionario[atributo])
      self.valoresXAtributo.append(tam)
    self.valoresXAtributo.append(1) # Para la clase
    self.longuitud_regla = sum(self.valoresXAtributo)

    # Para cada uno de los miembros de la poblacion
    for i in range(self.tamanio_poblacion):
      individuo = []

      # Generamos de forma aleatoria el numero de reglas que va a tener
      num_reglas = random.randint(1, self.maximo_reglas_individuo)

      # Para cada una de esas reglas
      for j in range(num_reglas):
        regla = np.zeros(self.longuitud_regla,dtype=int)

        # Para cada uno de los bits de cada regla
        for k in range(self.longuitud_regla):
          # Generamos un numero aleatorio entre 0 y 1
          x = random.randint(0,1)
          regla[k] = x

        # Comprobamos que la nueva regla no sea todo 1's o 0's
        if(all(regla) == 1 or any(regla) == 0):

          # Mutamos un bit aleatorio para que la regla sea valida
          pto_aleatorio= random.randint(0,self.longuitud_regla-1)
          regla[pto_aleatorio] = regla[pto_aleatorio] ^ 1

        
        individuo.append(regla)

      self.poblacion.append(individuo)


  def cruce_intra(self, progenitor1=None, progenitor2=None):

    random.seed() # Plantamos la semilla necesaria para el uso de rand

    # Comprobamos la probabilidad de cruce
    probabilidad = random.random()

    if(probabilidad <= self.prob_cruce):
      
      # Sacamos el numero de reglas del progenitor 1 y escogemos aleatoriamente una 
      # regla para el progenitor 1  
      num_reglas_p1 = len(progenitor1)
      rand_reglas_p1 = random.randint(0,num_reglas_p1-1)

      # Sacamos el numero de reglas del progenitor 2 y escogemos aleatoriamente una 
      # regla para el progenitor 2
      num_reglas_p2 = len(progenitor2)
      rand_reglas_p2 = random.randint(0,num_reglas_p2-1)

      # Sacamos un punto de cruce aleatorio
      pto_cruce = random.randint(0,self.longuitud_regla-1)

        
      copia_progenitor1 = progenitor1[rand_reglas_p1][pto_cruce:].copy()
        
      # Procedemos a cruzar en el progenitor1 en la regla concreta, la parte del progenitor 2 
      # correspondiente y viceversa
      progenitor1[rand_reglas_p1][pto_cruce:] = progenitor2[rand_reglas_p2][pto_cruce:]
      progenitor2[rand_reglas_p2][pto_cruce:] = copia_progenitor1



  def cruce_inter(self, progenitor1=None, progenitor2=None):

    random.seed() # Plantamos la semilla necesaria para el uso de rand

    # Comprobamos la probabilidad de cruce
    probabilidad = random.random()

    if(probabilidad <= self.prob_cruce):
      
      # Sacamos el numero de reglas del progenitor 1 y escogemos aleatoriamente una regla para el progenitor 1
      num_reglas_p1 = len(progenitor1)
      rand_reglas_p1 = random.randint(0,num_reglas_p1-1)

      # Sacamos el numero de reglas del progenitor 2 y escogemos aleatoriamente una regla para el progenitor 2
      num_reglas_p2 = len(progenitor2)
      rand_reglas_p2 = random.randint(0,num_reglas_p2-1)

      copia_progenitor1 = progenitor1[rand_reglas_p1].copy()

      # Cambiamos en el progenitor 1, la regla del progenitor 2
      progenitor1[rand_reglas_p1] = progenitor2[rand_reglas_p2]

      # Cambiamos en el progenitor 2, la regla del progenitor 1
      progenitor2[rand_reglas_p2] = copia_progenitor1

      
  def mutacion_bitflip(self, progenitor=None):

    random.seed() # Plantamos la semilla necesaria para el uso de rand
    

    # Sacamos el numero de reglas del progenitor y escogemos una al azar
    num_reglas_p1 = len(progenitor)
    rand_reglas_p1 = random.randint(0,num_reglas_p1-1)

# Recorremos la regla cambiando aquellos bits que consideremos
    for i in range(len(progenitor[rand_reglas_p1])):

      # Comprobamos la probabilidad de mutacion por cada bit
      if(random.random() <= self.prob_mutacion):

        # Cambiamos el bit (bitflip)
        progenitor[rand_reglas_p1][i] = progenitor[rand_reglas_p1][i] ^ 1 # Exclusive OR

    # Comprobamos que la regla mutada no sea todo 1's o 0's
    if(all(progenitor[rand_reglas_p1]) == 1 or any(progenitor[rand_reglas_p1]) == 0):

      # Mutamos un bit aleatorio para que la regla sea valida
      pto_aleatorio= random.randint(0,self.longuitud_regla-1)
      progenitor[rand_reglas_p1][pto_aleatorio] = progenitor[rand_reglas_p1][pto_aleatorio] ^ 1


  def mutacion_add_regla(self, progenitor=None):
    random.seed() # Plantamos la semilla necesaria para el uso de rand

    # Sacamos el numero de reglas del progenitor
    num_reglas_p1 = len(progenitor)

    # Si el progenitor ya tiene el numero maximo de reglas, no se muta
    if(num_reglas_p1 < self.maximo_reglas_individuo):

      # Comprobamos la probabilidad de mutacion del individuo
      if(random.random() <= self.prob_mutacion):
        
        # Creamos la nueva regla, inicializandola toda a 0's
        regla = np.zeros(self.longuitud_regla,dtype=int)

        # Para cada uno de los bits de cada regla
        for k in range(self.longuitud_regla):
          # Generamos un numero aleatorio entre 0 y 1
          x = random.randint(0,1)
          regla[k] = x

        # Comprobamos que la nueva regla no sea todo 1's o 0's
        if(all(regla) == 1 or any(regla) == 0):

          # Mutamos un bit aleatorio para que la regla sea valida
          pto_aleatorio= random.randint(0,self.longuitud_regla-1)
          regla[pto_aleatorio] = regla[pto_aleatorio] ^ 1   

        # Aniadimos la regla nueva al progenitor
        progenitor.append(regla)  


  def mutacion_remove_regla(self, progenitor=None):
    random.seed() # Plantamos la semilla necesaria para el uso de rand

    # Sacamos el numero de reglas del progenitor
    num_reglas_p1 = len(progenitor)

    # Si el progenitor tiene al menos dos reglas
    if(num_reglas_p1 >= 2): 
      # Comprobamos la probabilidad de mutacion del individuo
      if(random.random() <= self.prob_mutacion):

        # Elegimos una regla al azar
        rand_reglas_p1 = random.randint(0,num_reglas_p1-1)
        progenitor.remove(progenitor[rand_reglas_p1])
  

  def fitness(self, datos, individuo):
    atributos = self.valoresXAtributo[:-1]

    aciertos = 0            # Numero de veces que una regla del individuo acierta
    
    # Para cada uno de los datos de train
    for dato in datos:
      predice = []

      # Para cada una de las reglas del individuo
      for regla in individuo:
        
        index_min = 0
        activa = 1
        # Para cada uno de los atributos de la regla
        for atr in atributos:
          index_max = index_min + atr

          # Si no se activa ese atributo, salimos del bucle (No se activa la regla)
          if(not any(np.logical_and(regla[index_min:index_max], dato[index_min:index_max]))):
            activa = 0
            break
          else:
            index_min = index_min + atr

        # Si la regla se ha activado
        if activa:
          # Guardamos su prediccion
          predice.append(regla[-1])

      # Comprobamos que clase predice mayoritariamente
      if predice:
        clase_0 = predice.count(0)
        clase_1 = predice.count(1)

        # Si se predice clase 0
        if clase_0 > clase_1:
          if dato[-1] == 0:      
            aciertos = aciertos + 1
        # Si se predice clase 1
        elif clase_0 < clase_1:
          if dato[-1] == 1:
            aciertos = aciertos + 1
        
    return float(aciertos)/float(len(datos))

  def SeleccionProgenitores(self, datos):
    resultados = []
    
    # Para cada uno de los individuos de la poblacion
    for i in range(len(self.poblacion)):
      fitness = self.fitness(datos, self.poblacion[i])
      resultados.append([fitness, i])

    # Ordenamos la lista de mayor a menor fitness
    resultados.sort(reverse=True)
    # Devolvemos una lista con el indice y el fitness de cada individuo
    return resultados


  def entrenamiento(self, datos, datos_train_OneHot):

    # Generamos la poblacion inicial
    self.inizializarPoblacion(datos)

    grafica_media_fitness = []      # Lista donde guardaremos los fitness medios
    grafica_mejor_fitness = []      # Lista donde guardaremos los mejores fitness

    for k in range(self.condicion_terminacion):         
      # Calculamos el fitness de los individuos
      fitness_individuos = self.SeleccionProgenitores(datos_train_OneHot)
      
      # Guardamos el mejor fitness de la ronda
      grafica_mejor_fitness.append(fitness_individuos[0][0])

      # Calculamos el fitness medio
      media = 0.0
      for fit in fitness_individuos:
        media = media + fit[0]

      grafica_media_fitness.append(media/len(fitness_individuos))  

      # Escogemos de forma elitista los mejores individuos
      num_mejores = math.ceil((float(self.elitismo)/100)*float(self.tamanio_poblacion)) # Estos no se mutaran ni cortaran
      
      valores_elitista = np.arange(0,num_mejores,1).tolist() # Aquellas posiciones elitistas
      valores_normales = np.arange(num_mejores,len(fitness_individuos),1).tolist() # Aquellas posiciones no elitistas

      # Opcion 0
      # Seleccionar 2
      while(len(fitness_individuos)>1): # Por si acaso son impares

        individuos_seleccionados = []
        i=0
        while(len(individuos_seleccionados) < 2):
          max = sum(fitness_aux[0] for fitness_aux in fitness_individuos)
          pick = random.uniform(0, max)
          current=0
          flag=0
          i+=1
          for fitness_aux in fitness_individuos:
            current += fitness_aux[0]
            if(current > pick):
              individuos_seleccionados.append(fitness_aux[1])
              flag=1
              break
          if(flag==1): # Encontrado aquel que cumple la condicion
            fitness_individuos.remove(fitness_aux)

          # En el caso de no dar con ninguno que satisfaga la condicion previa, escogemos los que sean
          # Para ello estaremos en la segunda vuelta
          if(i==2 and len(individuos_seleccionados) < 2): # Segunda vuelta y ninguno cumple la condicion, el primero
            if(len(individuos_seleccionados) == 1):
              individuos_seleccionados.append(fitness_individuos[0][1])
              fitness_individuos.remove(fitness_individuos[0])
            elif(len(individuos_seleccionados) == 0):
              individuos_seleccionados.append(fitness_individuos[0][1])
              fitness_individuos.remove(fitness_individuos[0])
              individuos_seleccionados.append(fitness_individuos[0][1])
              fitness_individuos.remove(fitness_individuos[0])
          

        individuo1 = self.poblacion[individuos_seleccionados[0]]
        individuo2 = self.poblacion[individuos_seleccionados[1]]

        self.cruce(individuo1, individuo2)

        self.mutacion(individuo1)
        self.mutacion(individuo2)

      
    # Devolvemos el individuo con mayor fitness    
    fitness_individuos = self.SeleccionProgenitores(datos_train_OneHot)

    # Guardamos el mejor fitness de la ultima ronda
    grafica_mejor_fitness.append(fitness_individuos[0][0])

    # Calculamos el fitness medio
    media = 0.0
    for fit in fitness_individuos:
      media = media + fit[0]

    grafica_media_fitness.append(media/len(fitness_individuos))

    return self.poblacion[fitness_individuos[0][1]], grafica_media_fitness, grafica_mejor_fitness
    

  def clasifica(self, individuo, datos_test_OneHot):
    atributos = self.valoresXAtributo[:-1]
    predicciones = []
    
    # Para cada uno de los ejemplos de test
    for test in datos_test_OneHot:
      clases = []

      # Para cada una de las reglas del individuo
      for regla in individuo:
        
        index_min = 0
        activa = 1

        # Para cada uno de los atributos de la regla
        for atr in atributos:
          
          index_max = index_min + atr
          # Si no se activa ese atributo, salimos del bucle (No se activa la regla)
          if(not any(np.logical_and(regla[index_min:index_max], test[index_min:index_max]))):
            activa = 0
            break
          else:
            index_min = index_min + atr

        # Si la regla se ha activado
        if activa:
          clases.append(regla[-1])

      # Si no se ha activado ninguna regla, se predice por defecto 0
      if not clases:
        clases.append(0)
      
      # Seleccionamos la prediccion mas abundante
      # Predice la clase que mas se repita
      predice = max(set(clases), key=clases.count)
      predicciones.append(predice)

    return predicciones


  def plotear_graficas(self, media, mejores, fileName):

    n_epocas = len(media)

    plt.figure()

    # Limitamos las escalas
    plt.xlim([1, n_epocas])
    plt.ylim([0.0, 1.0])

    # Generamos los indices de la grafica
    index = []
    for i in range(n_epocas):
      index.append(i+1)

    plt.plot(index, media, "b")
    plt.plot(index, mejores, "r")

    # Aniadimos las etiquetas de los ejes
    plt.xlabel('Condicion de Terminacion')
    plt.ylabel('Fitness')

    # Nombre de la grafica
    plt.title('Grafica Fitness de ' + fileName[15:])
    
    # Leyenda
    plt.legend(["Media", "Mejores"], loc="lower right")
    plt.show()

