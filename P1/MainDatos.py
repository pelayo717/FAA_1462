from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorNaiveBayes,Clasificador
import sys

"""from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB"""

if __name__ == "__main__":

    ############################## TIC TAC TOE ###########################################

    fileName = "ConjuntosDatos/tic-tac-toe.data"
    datos_tic = Datos(fileName)
    
    # Probamos con 20 porciento y 10 iteraciones (Validacion Simple)
    validacion_simple_tic = ValidacionSimple(20,10)
    aux_simple_tic = validacion_simple_tic.creaParticiones(datos_tic)

    # Probamos con 10 k-iteraciones
    validacion_cruzada_tic = ValidacionCruzada(10)
    aux_cruzada_tic = validacion_cruzada_tic.creaParticiones(datos_tic)

    Clasificador_tic = ClasificadorNaiveBayes()
    media = Clasificador_tic.validacion(validacion_simple_tic,datos_tic)
    print("Error medio Simple Tic-Tac-Toe.data: " + str(media))

    media = Clasificador_tic.validacion(validacion_cruzada_tic,datos_tic)
    print("Error medio Cruzada Tic-Tac-Toe.data: " + str(media))

    ############################## GERMAN DATA ###########################################
    
    fileName = "ConjuntosDatos/german.data"
    datos_ger = Datos(fileName)
    
    # Probamos con 20 porciento y 10 iteraciones (Validacion Simple)
    validacion_simple_ger = ValidacionSimple(20,10)
    aux_simple_ger = validacion_simple_ger.creaParticiones(datos_ger)

    # Probamos con 10 k-iteraciones
    validacion_cruzada_ger = ValidacionCruzada(10)
    aux_cruzada_ger = validacion_cruzada_ger.creaParticiones(datos_ger)

    Clasificador_ger = ClasificadorNaiveBayes()
    media = Clasificador_ger.validacion(validacion_simple_ger,datos_ger)
    print("Error medio Simple German.data: " + str(media))

    media = Clasificador_ger.validacion(validacion_cruzada_ger,datos_ger)
    print("Error medio Cruzada German.data: " + str(media))




    ############################## SKLEARN ###########################################
    #X = load_iris()
    #x_train, x_test = train_test_split(X, test_size=0.2, random_state=0)
    #gnb = GaussianNB()

    #pred = gnb.fit(x_train).predict(x_test)
    #print(pred)


    
