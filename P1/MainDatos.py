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
    
    # Probamos con 75 porciento y 10 iteraciones (Validacion Simple)
    validacion_simple_tic = ValidacionSimple(75,10)
    aux_simple_tic = validacion_simple_tic.creaParticiones(datos_tic)

    # Probamos con 10 k-iteraciones
    validacion_cruzada_tic = ValidacionCruzada(10)
    aux_cruzada_tic = validacion_cruzada_tic.creaParticiones(datos_tic)

    Clasificador = ClasificadorNaiveBayes()
    media_error1, media_tp1, media_fp1, media_tn1, media_fn1 = Clasificador.validacion(validacion_simple_tic,datos_tic,True)
    print("Error medio Simple Tic-Tac-Toe.data: " + str(media_error1))

    media_error2, media_tp2, media_fp2, media_tn2, media_fn2 = Clasificador.validacion(validacion_cruzada_tic,datos_tic,True)
    print("Error medio Cruzada Tic-Tac-Toe.data: " + str(media_error2))

    ############################## GERMAN DATA ###########################################
    
    fileName = "ConjuntosDatos/german.data"
    datos_ger = Datos(fileName)
    
    # Probamos con 75 porciento y 10 iteraciones (Validacion Simple)
    validacion_simple_ger = ValidacionSimple(75,10)
    aux_simple_ger = validacion_simple_ger.creaParticiones(datos_ger)

    # Probamos con 10 k-iteraciones
    validacion_cruzada_ger = ValidacionCruzada(10)
    aux_cruzada_ger = validacion_cruzada_ger.creaParticiones(datos_ger)

    media_error3, media_tp3, media_fp3, media_tn3, media_fn3 = Clasificador.validacion(validacion_simple_ger,datos_ger,True)
    print("Error medio Simple German.data: " + str(media_error3))

    media_error4, media_tp4, media_fp4, media_tn4, media_fn4 = Clasificador.validacion(validacion_cruzada_ger,datos_ger,True)
    print("Error medio Cruzada German.data: " + str(media_error4))




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
    #Validacion Cruzada/ Con Preprocesado/ 10 Carpetas
    vc3 = Verificador_Multinominal(prepro=True,tipo_validacion=2,porcentaje=0.75,folds=10,archivo=fileName,alpha=1.0,fit_prior=True)

    ############################## MATRIZ CONFUSION ###########################################

    # Calculamos la media de las tasas de val. simple y val. cruzada para la matriz de confusion media
    mx1 = MatrizConfusion()

    # TIC-TAC-TOE
    print("\nTic-Tac-Toe")
    tpr, fpr = mx1.matrix_media(media_tp1, media_tp2, media_fp1, media_fp2, 
                    media_tn1, media_tn2, media_fn1, media_fn2)
    plot_points = [[fpr, tpr, 'NB']]
    mx1.plot(plot_points, "tic-tac-toe")

    # GERMAN
    print("\nGerman")

    tpr, fpr = mx1.matrix_media(media_tp3, media_tp4, media_fp3, media_fp4, 
                    media_tn3, media_tn4, media_fn3, media_fn4)
    plot_points = [[fpr, tpr, 'NB']]
    mx1.plot(plot_points, "german")