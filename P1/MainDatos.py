from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorNaiveBayes,Clasificador
from Verificador import Verificador_GaussianNB, Verificador_Multinominal
from MatrizConfusion import MatrizConfusion
import sys

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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
    media,datos_completos1,predicciones1 = Clasificador_tic.validacion(validacion_simple_tic,datos_tic)
    print("Error medio Simple Tic-Tac-Toe.data: " + str(media))

    media,datos_completos2,predicciones2 = Clasificador_tic.validacion(validacion_cruzada_tic,datos_tic)
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
    media,datos_completos3,predicciones3 = Clasificador_ger.validacion(validacion_simple_ger,datos_ger)
    print("Error medio Simple German.data: " + str(media))

    media,datos_completos4,predicciones4 = Clasificador_ger.validacion(validacion_cruzada_ger,datos_ger)
    print("Error medio Cruzada German.data: " + str(media))




    ############################## SKLEARN ###########################################
    fileName = "ConjuntosDatos/tic-tac-toe.data"
    #Validacion Simple/ Sin Preprocesado/ 75% 
    vg = Verificador_GaussianNB(prepro=False,tipo_validacion=1,porcentaje=0.75,folds=3,archivo=fileName)
    #Validacion Simple/ Con Preprocesado/ 75% 
    vg1 = Verificador_GaussianNB(prepro=True,tipo_validacion=1,porcentaje=0.75,folds=3,archivo=fileName)
    #Validacion Cruzada/ Sin Preprocesado/ 10 Carpetas
    vg2 = Verificador_GaussianNB(prepro=False,tipo_validacion=2,porcentaje=0.75,folds=10,archivo=fileName)
    #Validacion Cruzada/ Con Preprocesado/ 10 Carpetas
    vg3 = Verificador_GaussianNB(prepro=True,tipo_validacion=2,porcentaje=0.75,folds=10,archivo=fileName)

    #Validacion Simple/ Sin Preprocesado/ 75% 
    vc = Verificador_Multinominal(prepro=False,tipo_validacion=1,porcentaje=0.75,folds=3,archivo=fileName,alpha=1.0,fit_prior=True)
    #Validacion Simple/ Con Preprocesado/ 75% 
    vc1 = Verificador_Multinominal(prepro=True,tipo_validacion=1,porcentaje=0.75,folds=3,archivo=fileName,alpha=1.0,fit_prior=True)
    #Validacion Cruzada/ Sin Preprocesado/ 10 Carpetas
    vc2 = Verificador_Multinominal(prepro=False,tipo_validacion=2,porcentaje=0.75,folds=10,archivo=fileName,alpha=1.0,fit_prior=True)
    #Validacion Cruzada/ Sin Preprocesado/ 10 Carpetas
    vc3 = Verificador_Multinominal(prepro=True,tipo_validacion=2,porcentaje=0.75,folds=10,archivo=fileName,alpha=1.0,fit_prior=True)

    ############################## MATRIZ CONFUSION ###########################################


    mx1 = MatrizConfusion(predicciones2,datos_completos2)
    mx1.matrix_design()
    mx1.plot()

