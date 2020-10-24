from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from EstrategiaParticionado import ValidacionCruzada
from Clasificador import ClasificadorNaiveBayes,Clasificador
from Verificador import Verificador_GaussianNB, Verificador_Multinominal
from MatrizConfusion import MatrizConfusion

import sys
from tabulate import tabulate

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
    validacion_cruzada_tic = ValidacionCruzada(6)
    aux_cruzada_tic = validacion_cruzada_tic.creaParticiones(datos_tic)

    Clasificador = ClasificadorNaiveBayes()
    media_error1, media_tp1, media_fp1, media_tn1, media_fn1 = Clasificador.validacion(validacion_simple_tic,datos_tic,True)
    media_error2, media_tp2, media_fp2, media_tn2, media_fn2 = Clasificador.validacion(validacion_cruzada_tic,datos_tic,True)

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
    media_error4, media_tp4, media_fp4, media_tn4, media_fn4 = Clasificador.validacion(validacion_cruzada_ger,datos_ger,True)
    
    # Impresion de los resultados
    resultados = [[round(media_error1, 3), round(media_error3, 3)], [round(media_error2, 3), 
    round(media_error4, 3)]]


    ############################## SKLEARN ###########################################
    vgNB = Verificador_GaussianNB()
    # GaussianNB => Sin Preprocesado
    #     TIC-TAC-TOE => Validacion Simple (TRAIN=0.75)
    tic_simple_sin = vgNB.clasificate(prepro=False,tipo_validacion=1,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/tic-tac-toe.data")
    #     TIC-TAC-TOE => Validacion Cruzada (KFOLDS = 6)
    tic_cruzada_sin = vgNB.clasificate(prepro=False,tipo_validacion=2,porcentaje=0.75,folds=6,archivo="ConjuntosDatos/tic-tac-toe.data")

    #     GERMAN => Validacion Simple (TRAIN=0.75)
    german_simple_sin = vgNB.clasificate(prepro=False,tipo_validacion=1,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/german.data")
    #     GERMAN => Validacion Cruzada (KFOLDS = 6)
    german_cruzada_sin = vgNB.clasificate(prepro=False,tipo_validacion=2,porcentaje=0.75,folds=6,archivo="ConjuntosDatos/german.data")

    # Impresion de los resultados
    resultados_sk_sin = [[round(tic_simple_sin, 3), round(german_simple_sin, 3)], [round(tic_cruzada_sin, 3), round(german_cruzada_sin, 3)]]

    # GaussianNB => Con Preprocesado

    #     TIC-TAC-TOE => Validacion Simple (TRAIN=0.75)
    tic_simple_con = vgNB.clasificate(prepro=True,tipo_validacion=1,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/tic-tac-toe.data")
    #     TIC-TAC-TOE => Validacion Cruzada (KFOLDS = 6)
    tic_cruzada_con = vgNB.clasificate(prepro=True,tipo_validacion=2,porcentaje=0.75,folds=6,archivo="ConjuntosDatos/tic-tac-toe.data")

    #     GERMAN => Validacion Simple (TRAIN=0.75)
    german_simple_con = vgNB.clasificate(prepro=True,tipo_validacion=1,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/german.data")
    #     GERMAN => Validacion Cruzada (KFOLDS = 6)
    german_cruzada_con = vgNB.clasificate(prepro=True,tipo_validacion=2,porcentaje=0.75,folds=6,archivo="ConjuntosDatos/german.data")

    # Impresion de los resultados
    resultados_sk_con = [[round(tic_simple_con, 3), round(german_simple_con, 3)], [round(tic_cruzada_con, 3), round(german_cruzada_con, 3)]]

    # Impresion de las tablas
    print("Practica 1:")
    print(tabulate(resultados, headers=['Tasa de error', 'Tic-Tac-Toe', 'German'], showindex=['Val. Simple', 'Val. Cruzada'], tablefmt='fancy_grid'))

    print("SKLearn:")
    print("Sin Preprocesado")
    print(tabulate(resultados_sk_sin, headers=['Tasa de error', 'Tic-Tac-Toe', 'German'], showindex=['Val. Simple', 'Val. Cruzada'], tablefmt='fancy_grid'))
    print("Con Preprocesado")
    print(tabulate(resultados_sk_con, headers=['Tasa de error', 'Tic-Tac-Toe','German'], showindex=['Val. Simple', 'Val. Cruzada'], tablefmt='fancy_grid'))

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