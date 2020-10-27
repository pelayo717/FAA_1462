from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from Clasificador import *
from Distancias import *

if __name__ == "__main__":

    fileName = "ConjuntosDatos/pima-indians-diabetes.data"
    datos = Datos(fileName)

    validacion_simple = ValidacionSimple(75,10)
    aux_simple = validacion_simple.creaParticiones(datos)

    knn = ClasificadorVecinosProximos(3, "Euclidea")

    knn.entrenamiento(datos)
    print(knn.clasifica(datos, aux_simple[0].indicesTrain, aux_simple[0].indicesTest))