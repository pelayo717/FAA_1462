from abc import ABCMeta,abstractmethod
import random
import math

class Particion():

  # Esta clase mantiene la lista de indices de Train y Test para cada particion del conjunto de particiones  
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  error = None        # Porcentaje de error del modelo

  @abstractmethod
  def creaParticiones(self,datos,seed=None):
    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):
  
  porcentaje = None       # Porcentaje de la muestra que se utilizara como entrenamiento
  numEjecuciones = None   # Numero de ejecuciones de la validacion simple

  def __init__(self, por, num):
    self.porcentaje = por
    self.numEjecuciones = num

  
  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el numero de ejecuciones deseado
  # Devuelve una lista de particiones (clase Particion)
  def creaParticiones(self, datos, seed=None):
    
    # Control de errores por si el parametro seed es 'None'
    if seed is None:
      seed = 10

    # Generamos la semilla
    random.seed(seed) 
    listaParticiones = [] # Lista de tipo Particion con las particiones generadas

    # Cantidad de datos de la muestra que se cogeran como datos de testeo
    numTest = int(datos.cantidadDatos * (1 - ( float(self.porcentaje / float(100)))))

    # Creamos una particion nueva por cada ejecucion y la aniadimos a la lista
    for i in range(self.numEjecuciones):
      particion = Particion()
      
      # Metemos todos los indices posibles en train. Generamos indices aleatorios y los metemos en test sacandolos de train
      # De esta forma podemos comprobar si ya se han repetido y los restantes de train ya se encuentran en test
      particion.indicesTrain = list(range(0, datos.cantidadDatos))

      # Insertamos todos los indices generados aleatoriamente en Test y los eliminamos de Train
      for j in range(numTest):
        
        encontrado = False        # Flag que garantiza que el indice generado no este repetido
        while encontrado is False: # De no haber sido extraido previamente, llevamos a cabo la extraccion y la insercion de cada lista
          indice = random.randint(0, (datos.cantidadDatos - 1))
          # Comprobamos si se encuentra todavia en Train
          if indice in particion.indicesTrain:
            particion.indicesTest.append(indice)
            particion.indicesTrain.remove(indice)
            encontrado = True
      
      # Aniadimos la particion creada, al listado de particiones a retornar
      listaParticiones.append(particion)
      particion = None

    return listaParticiones
      
      
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):
  
  numParticiones = None # Numero de particiones que se usaran como entrenamiento (k-1) y como test (1)

  def __init__(self, num):
    self.numParticiones = num

  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  def creaParticiones(self,datos,seed=None):

    # Control de errores por si el parametro seed es 'None'
    if seed is None:
      seed = 10

    # Generamos la semilla   
    random.seed(seed)
    listaParticiones = [] # Lista de tipo Particion con las particiones generadas

    # Numero total de datos que encontramos en el dataset
    numDatos = datos.cantidadDatos

    # Redondeamos el numero de valores por bloque hacia el techo
    # Por tanto, en el caso de que el reparto de datos no sea un numero exacto, 
    # habra un bloque que presente menos datos que los demas
    numDatosPorBloques = int(math.ceil(float(numDatos)/float(self.numParticiones)))

    #================= CREACION DE BLOQUES ============#
    lista_datos = list(range(0, datos.cantidadDatos))
    
    # Contendra todas las particiones de cada iteracion. Solo los obtenemos la primera vez
    bloques = []
    for i in range(self.numParticiones):
      aux_bloque = []

      # Rellenamos un bloque con el numero de indices que le corresponde 
      for j in range(numDatosPorBloques):

        # Mientras halla datos en la lista, vamos insertando en cada bloque 
        if (len(lista_datos) > 0):
          encontrado = False        # Flag que garantiza que el indice generado no este repetido

          while encontrado is False: # Comenzamos la extraccion de datos de la lista general
            indice = random.randint(0, (datos.cantidadDatos - 1)) # Escogemos aleatoriamente el indice de los datos totales
            if indice in lista_datos: # Comprobamos si se encuentra todavia en lista_datos
              aux_bloque.append(indice) # De ser asi, se extrae de la lista general, y se incluye en el bloque concreto
              lista_datos.remove(indice) # Sino, se seguiran buscando la cantidad de datos necesarios hasta completar el bloque
              encontrado = True
        else: # Condicion que se da cuando seguimos necesitando numeros para rellenar el bloque pero no quedan mas en la lista 
          break #Paramos la ejecucion y damos por acabado los repartos

      bloques.append(aux_bloque) #Aniadimos el bloque a la lista a retornar
      aux_bloque = None

    #====================== ASIGNACION DE BLOQUES ===================#

    # Debemos realizar k iteraciones, y por cada iteracion debemos dividir los datos en k subconjuntos.
    # De estos subconjuntos 1, los usaremos de prueba mientras el resto seran de entrenamiento
    for i in range(self.numParticiones):
      particion = Particion()

      # Rellenamos la particion con los bloques train y el test
      for j in range(len(bloques)):
        # Bloque de test
        if(j==i): # La idea es extraer de cada bloque los indices que alberga, al ser test, solo obtendremos de un bloque
          for k in range(len(bloques[j])):
            particion.indicesTest.append(bloques[j][k])
        else: # Pero los de train, se juntaran en el array indicesTrain, de la particion concreta
          for k in range(len(bloques[j])):
            particion.indicesTrain.append(bloques[j][k])

      listaParticiones.append(particion) # Aniadimos la particion concreta y continuamos hasta obtener todas las requeridas
      particion=None
    
    return listaParticiones
    
