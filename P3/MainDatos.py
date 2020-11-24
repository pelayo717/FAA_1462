from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
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

    