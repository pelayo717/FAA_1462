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
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta.
  #  Se pasan en el constructor 
  
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
    for i in range( self.numEjecuciones ):
      particion = Particion()
      
      # Metemos todos los indices posibles en train. Generamos indices aleatorios y los metemos en test sacandolos de train.
      # De esta forma podemos comprobar si ya se han repetido y los restantes de train ya se encuentran en test.
      particion.indicesTrain = list(range(0, datos.cantidadDatos))

      # Insertamos todos los indices generados aleatoriamente en Test y los eliminamos de Train
      for j in range( numTest ):
        
        encontrado = False        # Flag que garantiza que el indice generado no este repetido
        while encontrado is False:

          indice = random.randint(0, (datos.cantidadDatos - 1))
          # Comprobamos si se encuentra todavia en Train
          if indice in particion.indicesTrain:
            particion.indicesTest.append(indice)
            particion.indicesTrain.remove(indice)
            encontrado = True
      
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
    # Por tanto, en el caso de que el reparto de datos no sea un numero clavado, 
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
          while encontrado is False:
            indice = random.randint(0, (datos.cantidadDatos - 1)) # Escogemos aleatoriamente el indice de los datos totales
            if indice in lista_datos: # Comprobamos si se encuentra todavia en lista_datos
              aux_bloque.append(indice)
              lista_datos.remove(indice)
              encontrado = True
        else: # Condicion que se da cuando seguimos necesitando numeros para rellenar el bloque pero no quedan mas en la lista 
          break

      bloques.append(aux_bloque)
      aux_bloque = None

    #====================== ASIGNACION DE BLOQUES ===================#

    # Debemos realizar k iteraciones, y por cada iteracion debemos dividir los datos en k subconjuntos.
    # De estos subconjuntos 1 los usaremos de prueba mientras el resto seran de entrenamiento
    for i in range(self.numParticiones):
      particion = Particion()

      # Rellenamos la particion con los bloques train y el test
      for j in range(len(bloques)):
        # Bloque de test
        if(j==i): #No es igualar, es incluir, si igualas, sobreescribes los otros XD
          for k in range(len(bloques[j])):
            particion.indicesTest.append(bloques[j][k])
        else: # Resto, bloques de train
          for k in range(len(bloques[j])):
            particion.indicesTrain.append(bloques[j][k])

      listaParticiones.append(particion)
      particion=None
    
    return listaParticiones
    
