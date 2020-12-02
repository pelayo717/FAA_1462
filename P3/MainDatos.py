from Datos import Datos
from EstrategiaParticionado import *
from Clasificador import *
from Distancias import *
from Verificador import *
from MatrizConfusion import MatrizConfusion
from tabulate import tabulate

if __name__ == "__main__":

    # Abrimos los ficheros y extraemos los datos
    fileName = "ConjuntosDatos/titanic.data"
    datos_titanic = Datos(fileName)


    # Creamos las validaciones
    validacion_simple = ValidacionSimple(75,10)

    validacion_cruzada = ValidacionCruzada(6)

    ag = ClasficadorAlgoritmoGenetico(10, 100, 5)

    medias = ag.validacion(validacion_simple, datos_titanic)

    aux1 = ag.poblacion[0]

    aux2 = ag.poblacion[1]

    ag.cruce("intra",aux1,aux2)

    ag.mutacion("borrar",None,0.5,aux1)
    