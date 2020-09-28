from abc import ABCMeta,abstractmethod


class Particion():

  # Esta clase mantiene la lista de indices de Train y Test para cada particion del conjunto de particiones  
  def __init__(self):
    self.indicesTrain=[]
    self.indicesTest=[]

#####################################################################################################

class EstrategiaParticionado:
  
  # Clase abstracta
  __metaclass__ = ABCMeta
  
  # Atributos: deben rellenarse adecuadamente para cada estrategia concreta. Se pasan en el constructor 
  
  error = None        # Porcentaje de error del modelo

  @abstractmethod
  # TODO: esta funcion deben ser implementadas en cada estrategia concreta  
  def creaParticiones(self,datos,seed=None):
    pass
  

#####################################################################################################

class ValidacionSimple(EstrategiaParticionado):
  
  porcentaje = None       # Porcentaje de la muestra que se utilizara como entrenamiento
  numEjecuciones = None   # Numero de ejecuciones de la validación simple

  def __init__(self, por, num):
    self.porcentaje = por
    self.numEjecuciones = num

  
  # Crea particiones segun el metodo tradicional de division de los datos segun el porcentaje deseado y el n�mero de ejecuciones deseado
  # Devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self, datos, seed=None):
    # Control de errores por si el parametro seed es 'None'
    if seed is None:
      seed = 10

    #Generamos la semilla
    random.seed(seed) 
    listaParticiones = [] # Lista de tipo Particion con las particiones generadas

    # Cantidad de datos de la muestra que se cogeran como datos de testeo
    numTest = datos.cantidadDatos * (1 - ( self.porcentaje / 100 ))
    
    # Creamos una particion nueva por cada ejecucion y la aniadimos a la lista
    for i = 0 in range( self.numEjecuciones ):
      particion = Particion()
      
      # Metemos todos los indices posibles en train. Generamos indices aleatorios y los metemos en test sacandolos de train.
      # De esta forma podemos comprobar si ya se han repetido y los restantes de train ya se encuentran en test.
      particion.indicesTrain = list(range(0, datos.cantidadDatos))

      # Insertamos todos los indices generados aleatoriamente en Test y los eliminamos de Train
      for j = 0 in range( numTest ):
        
        encontrado = False        # Flag que garantiza que el indice generado no esté repetido
        while encontrado is False:

          indice = random.randint(0, (datos.cantidadDatos - 1))
          # Comprobamos si se encuentra todavia en Train
          if indice in particion.indicesTrain:
            particion.indicesTest.append(indice)
            particion.indicesTrain.remove(indice)
            encontrado = True
      
      listaParticiones.append(particion)

    return listaParticiones
      
      
#####################################################################################################      
class ValidacionCruzada(EstrategiaParticionado):
  
  # Crea particiones segun el metodo de validacion cruzada.
  # El conjunto de entrenamiento se crea con las nfolds-1 particiones y el de test con la particion restante
  # Esta funcion devuelve una lista de particiones (clase Particion)
  # TODO: implementar
  def creaParticiones(self,datos,seed=None):   
    random.seed(seed)
    pass
    
