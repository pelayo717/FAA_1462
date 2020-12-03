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

    ag = ClasficadorAlgoritmoGenetico( 
                        tam_poblacion=10, cond_terminacion=100, max_reglas=5, 
                        tipo_cruce = 0, tipo_mutacion=0, prob_cruce=0.25, 
                        prob_mutacion=0.05)

    medias = ag.validacion(validacion_simple, datos_titanic)

    aux1 = ag.poblacion[0]

    aux2 = ag.poblacion[1]

    ag.cruce_intra(aux1, aux2)

    ag.mutacion_bitflip(aux1)
    