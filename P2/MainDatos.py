from Datos import Datos
from Clasificador_KNN import Clasificador_KNN
from Normalizar import Normanlizar

if __name__ == "__main__":

    ############################## TIC TAC TOE ###########################################

    fileName = "ConjuntosDatos/pima-indians-diabetes.data"
    datos = Datos(fileName)

    aux = Normanlizar()
    aux.calcularMediasDesv(datos.datos,datos.nominalAtributos)
    aux.normalizarDatos(datos.datos,datos.nominalAtributos)

    cl = Clasificador_KNN(3)
    cl.Ecuclidea(datos,10)
    cl.Manhattan(datos,10)
