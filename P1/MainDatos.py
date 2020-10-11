from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorNaiveBayes,Clasificador
import sys

if __name__ == "__main__":
    fileName = "ConjuntosDatos/tic-tac-toe.data"
    datos = Datos(fileName)
    
    # Probamos con 20 porciento y 3 iteraciones (Validacion Simple)
    validacion_simple = ValidacionSimple(20,3)
    aux_simple = validacion_simple.creaParticiones(datos)

    # Probamos con 2 k-iteraciones
    validacion_cruzada = ValidacionCruzada(3)
    aux_cruzada = validacion_cruzada.creaParticiones(datos)

    """print(aux_cruzada)
    print(len(aux_cruzada[0].indicesTrain[0]))
    print(len(aux_cruzada[0].indicesTrain[1]))
    print(len(aux_cruzada[0].indicesTest[0]))
   
    print(aux_cruzada[0].indicesTrain[0])
    print(aux_cruzada[0].indicesTrain[1])
    print(aux_cruzada[0].indicesTest[0])"""

    Clasificador = ClasificadorNaiveBayes()
    media = Clasificador.validacion(validacion_simple,datos)
    print("Error medio: " + str(media))

    
