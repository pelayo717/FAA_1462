from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorNaiveBayes,Clasificador
import sys

"""from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB"""

if __name__ == "__main__":
    fileName = "ConjuntosDatos/german.data"
    datos = Datos(fileName)
    
    # Probamos con 20 porciento y 3 iteraciones (Validacion Simple)
    validacion_simple = ValidacionSimple(20,1)
    aux_simple = validacion_simple.creaParticiones(datos)

    # Probamos con 3 k-iteraciones
    validacion_cruzada = ValidacionCruzada(3)
    aux_cruzada = validacion_cruzada.creaParticiones(datos)

    Clasificador = ClasificadorNaiveBayes()
    media = Clasificador.validacion(validacion_simple,datos)
    print("Error medio Simple German.data: " + str(media))

    media = Clasificador.validacion(validacion_cruzada,datos)
    print("Error medio Cruzada German.data: " + str(media))

    #### SKLEARN ####
    #X = load_iris()
    #x_train, x_test = train_test_split(X, test_size=0.2, random_state=0)
    #gnb = GaussianNB()

    #pred = gnb.fit(x_train).predict(x_test)
    #print(pred)


    
