from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from Clasificador import *
from Distancias import *
from Verificador import *
from MatrizConfusion import MatrizConfusion
from tabulate import tabulate

if __name__ == "__main__":

    # Abrimos los ficheros y extraemos los datos
    fileName = "ConjuntosDatos/pima-indians-diabetes.data"
    datos_diabetes = Datos(fileName)

    fileName = "ConjuntosDatos/wdbc.data"
    datos_wdbc = Datos(fileName)

    # Creamos las validaciones
    validacion_simple_diabetes = ValidacionSimple(75,10)
    #simple_diabetes = validacion_simple_diabetes.creaParticiones(datos_diabetes)

    validacion_cruzada_diabetes = ValidacionCruzada(6)
    #cruzada_diabetes = validacion_cruzada_diabetes.creaParticiones(datos_diabetes)

    validacion_simple_wdbc = ValidacionSimple(75,10)
    #simple_wdbc = validacion_simple_wdbc.creaParticiones(datos_wdbc)

    validacion_cruzada_wdbc = ValidacionCruzada(6)
    #cruzada_wdbc = validacion_cruzada_wdbc.creaParticiones(datos_wdbc)

    knn = ClasificadorVecinosProximos(5, Euclidea)

    medias_simples_diabetes_knn_5 = knn.validacion(validacion_simple_diabetes, datos_diabetes,False,True)
    medias_cruzadas_diabetes_knn_5 = knn.validacion(validacion_cruzada_diabetes, datos_diabetes,False,True)

    medias_simples_wdbc_knn_5 = knn.validacion(validacion_simple_wdbc, datos_wdbc,False,True)
    medias_cruzadas_wdbc_knn_5 = knn.validacion(validacion_cruzada_wdbc, datos_wdbc,False,True)

    medias_simples_diabetes_knn_5_sin = knn.validacion(validacion_simple_diabetes, datos_diabetes,False,False)
    medias_cruzadas_diabetes_knn_5_sin = knn.validacion(validacion_cruzada_diabetes, datos_diabetes,False,False)

    medias_simples_wdbc_knn_5_sin = knn.validacion(validacion_simple_wdbc, datos_wdbc,False,False)
    medias_cruzadas_wdbc_knn_5_sin = knn.validacion(validacion_cruzada_wdbc, datos_wdbc,False,False)




    """############################## Diabetes ##############################
    validacion_simple_diabetes = ValidacionSimple(75,10)
    simple_diabetes = validacion_simple_diabetes.creaParticiones(datos_diabetes)

    validacion_cruzada_diabetes = ValidacionCruzada(6)
    cruzada_diabetes = validacion_cruzada_diabetes.creaParticiones(datos_diabetes)
    
    medias_simples_diabetes_nb = nb.validacion(validacion_simple_diabetes, datos_diabetes, True)
    medias_cruzadas_diabetes_nb = nb.validacion(validacion_cruzada_diabetes, datos_diabetes, True)

    medias_simples_diabetes_rl = cl.validacion(validacion_simple_diabetes, datos_diabetes)
    medias_cruzadas_diabetes_rl = cl.validacion(validacion_cruzada_diabetes, datos_diabetes)

    medias_simples_diabetes_knn = knn.validacion(validacion_simple_diabetes, datos_diabetes)
    medias_cruzadas_diabetes_knn = knn.validacion(validacion_cruzada_diabetes, datos_diabetes)
    
    

    ############################## Wdbc ##############################
    validacion_simple_wdbc = ValidacionSimple(75,10)
    simple_wdbc = validacion_simple_wdbc.creaParticiones(datos_wdbc)

    validacion_cruzada_wdbc = ValidacionCruzada(6)
    cruzada_wdbc = validacion_cruzada_wdbc.creaParticiones(datos_wdbc)

    medias_simples_wdbc_nb = nb.validacion(validacion_simple_wdbc, datos_wdbc, True)
    medias_cruzadas_wdbc_nb = nb.validacion(validacion_cruzada_wdbc, datos_wdbc, True)
    
    medias_simples_wdbc_rl = cl.validacion(validacion_simple_wdbc, datos_wdbc)
    medias_cruzadas_wdbc_rl = cl.validacion(validacion_cruzada_wdbc, datos_wdbc)
    
    medias_simples_wdbc_knn = knn.validacion(validacion_simple_wdbc, datos_wdbc)
    medias_cruzadas_wdbc_knn = knn.validacion(validacion_cruzada_wdbc, datos_wdbc)


    # Guardamos resultados de los resultados KNN
    resultados_P2 = [[round(medias_simples_diabetes_knn[0], 3), round(medias_simples_wdbc_knn[0], 3)], 
            [round(medias_cruzadas_diabetes_knn[0], 3), round(medias_cruzadas_wdbc_knn[0], 3)]]

    # Impresion de las tablas
    print("Practica 2 KNN:")
    print(tabulate(resultados_P2, headers=['Tasa de error', 'Diabetes', 'Wdbc'], 
        showindex=['Val. Simple', 'Val. Cruzada'], tablefmt='fancy_grid'))


    resultados_P2_rl = [[round(medias_simples_diabetes_rl[0], 3), round(medias_simples_wdbc_rl[0], 3)], 
            [round(medias_cruzadas_diabetes_rl[0], 3), round(medias_cruzadas_wdbc_rl[0], 3)]]

    print("\nPractica 2 Regresion Logistica:")
    print(tabulate(resultados_P2_rl, headers=['Tasa de error', 'Diabetes', 'Wdbc'], 
        showindex=['Val. Simple', 'Val. Cruzada'], tablefmt='fancy_grid'))

    ############################## MATRIZ CONFUSION ###########################################

    # Calculamos la media de las tasas de val. simple y val. cruzada para la matriz de confusion media
    mx1 = MatrizConfusion()

    # DIABETES

    print("\nDiabetes NB")
    tpr_nb, fpr_nb = mx1.matrix_media(medias_simples_diabetes_nb[1], 
        medias_cruzadas_diabetes_nb[1], 
        medias_simples_diabetes_nb[2], 
        medias_cruzadas_diabetes_nb[2],
        medias_simples_diabetes_nb[3], 
        medias_cruzadas_diabetes_nb[3],
        medias_simples_diabetes_nb[4], 
        medias_cruzadas_diabetes_nb[4])

    print("\nDiabetes KNN")
    tpr_knn, fpr_knn = mx1.matrix_media(medias_simples_diabetes_knn[1], 
        medias_cruzadas_diabetes_knn[1], 
        medias_simples_diabetes_knn[2], 
        medias_cruzadas_diabetes_knn[2],
        medias_simples_diabetes_knn[3], 
        medias_cruzadas_diabetes_knn[3],
        medias_simples_diabetes_knn[4], 
        medias_cruzadas_diabetes_knn[4])


    print("\nDiabetes Regresion Logistica")
    tpr_rl, fpr_rl = mx1.matrix_media(medias_simples_diabetes_rl[1], 
        medias_cruzadas_diabetes_rl[1], 
        medias_simples_diabetes_rl[2], 
        medias_cruzadas_diabetes_rl[2],
        medias_simples_diabetes_rl[3], 
        medias_cruzadas_diabetes_rl[3],
        medias_simples_diabetes_rl[4], 
        medias_cruzadas_diabetes_rl[4])

    plot_points = [[fpr_nb, tpr_nb, 'NB'], [fpr_knn, tpr_knn, 'KNN'], [fpr_rl, tpr_rl, 'RL']]
    mx1.plot(plot_points, "Diabetes")

    # WDBC

    print("\nWdbc NB")
    tpr_nb, fpr_nb = mx1.matrix_media(medias_simples_wdbc_nb[1], 
        medias_cruzadas_wdbc_nb[1], 
        medias_simples_wdbc_nb[2], 
        medias_cruzadas_wdbc_nb[2],
        medias_simples_wdbc_nb[3], 
        medias_cruzadas_wdbc_nb[3],
        medias_simples_wdbc_nb[4], 
        medias_cruzadas_wdbc_nb[4])

    
    print("\nWdbc KNN")
    tpr_knn, fpr_knn = mx1.matrix_media(medias_simples_wdbc_knn[1], 
        medias_cruzadas_wdbc_knn[1], 
        medias_simples_wdbc_knn[2], 
        medias_cruzadas_wdbc_knn[2],
        medias_simples_wdbc_knn[3], 
        medias_cruzadas_wdbc_knn[3],
        medias_simples_wdbc_knn[4], 
        medias_cruzadas_wdbc_knn[4])
    
    print("\nWdbc Regresion Logistica")
    tpr_rl, fpr_rl = mx1.matrix_media(medias_simples_wdbc_rl[1], 
        medias_cruzadas_wdbc_rl[1], 
        medias_simples_wdbc_rl[2], 
        medias_cruzadas_wdbc_rl[2],
        medias_simples_wdbc_rl[3], 
        medias_cruzadas_wdbc_rl[3],
        medias_simples_wdbc_rl[4], 
        medias_cruzadas_wdbc_rl[4])
    
    plot_points = [[fpr_nb, tpr_nb, 'NB'], [fpr_knn, tpr_knn, 'KNN'], [fpr_rl, tpr_rl, 'RL']]
    mx1.plot(plot_points, "Wdbc")

    #################### VERIFICADORES KNN y REGRESION LOGISTICA ###############

    verificador_knn = Verificados_KVecinos(3,'uniform','euclidean')


    diabetes_sin_kn_simple_eu = verificador_knn.clasificate(prepro=False,tipo_validacion=1,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/pima-indians-diabetes.data")

    diabetes_sin_kn_cruzada_eu = verificador_knn.clasificate(prepro=False,tipo_validacion=2,porcentaje=0.75,folds=6,archivo="ConjuntosDatos/pima-indians-diabetes.data")

    wdbc_sin_kn_simple_eu = verificador_knn.clasificate(prepro=False,tipo_validacion=1,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/wdbc.data")

    wdbc_sin_kn_cruzada_eu = verificador_knn.clasificate(prepro=False,tipo_validacion=2,porcentaje=0.75,folds=6,archivo="ConjuntosDatos/wdbc.data")

    
    verificador_knn = Verificados_KVecinos(3,'uniform','manhattan')

    diabetes_sin_kn_simple_man = verificador_knn.clasificate(prepro=False,tipo_validacion=1,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/pima-indians-diabetes.data")

    diabetes_sin_kn_cruzada_man = verificador_knn.clasificate(prepro=False,tipo_validacion=2,porcentaje=0.75,folds=6,archivo="ConjuntosDatos/pima-indians-diabetes.data")

    wdbc_sin_kn_simple_man = verificador_knn.clasificate(prepro=False,tipo_validacion=1,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/wdbc.data")

    wdbc_sin_kn_cruzada_man = verificador_knn.clasificate(prepro=False,tipo_validacion=2,porcentaje=0.75,folds=6,archivo="ConjuntosDatos/wdbc.data")

    verificador_knn = Verificados_KVecinos(3,'uniform','mahalanobis')

    diabetes_sin_kn_simple_mah = verificador_knn.clasificate(prepro=False,tipo_validacion=1,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/pima-indians-diabetes.data")

    diabetes_sin_kn_cruzada_mah = verificador_knn.clasificate(prepro=False,tipo_validacion=2,porcentaje=0.75,folds=6,archivo="ConjuntosDatos/pima-indians-diabetes.data")

    wdbc_sin_kn_simple_mah = verificador_knn.clasificate(prepro=False,tipo_validacion=1,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/wdbc.data")

    wdbc_sin_kn_cruzada_mah = verificador_knn.clasificate(prepro=False,tipo_validacion=2,porcentaje=0.75,folds=6,archivo="ConjuntosDatos/wdbc.data")


    resultados_sk_sin = [[round(diabetes_sin_kn_simple_eu, 3), round(wdbc_sin_kn_simple_eu, 3), 
        round(diabetes_sin_kn_simple_man,3), round(wdbc_sin_kn_simple_man,3), 
        round(diabetes_sin_kn_simple_mah,3), round(wdbc_sin_kn_simple_mah,3)],  
        [round(diabetes_sin_kn_cruzada_eu, 3), round(wdbc_sin_kn_cruzada_eu, 3), 
        round(diabetes_sin_kn_cruzada_man,3), round(wdbc_sin_kn_cruzada_man,3),
        round(diabetes_sin_kn_cruzada_mah,3), round(wdbc_sin_kn_cruzada_mah,3)]]

    print("SKLearn:")
    print("Sin Preprocesado KNN")
    print(tabulate(resultados_sk_sin, headers=['Tasa de error', 'Diabetes(Euclidean)', 'Wdbc(Euclidean)','Diabetes(Manhattan)','Wdbc(Manhattan)','Diabetes(Mahalanobis)','Wdbc(Mahalanobis)'], showindex=['Val. Simple', 'Val. Cruzada'], tablefmt='fancy_grid'))

    # La panlizacion se refiere a Lasso 1 o 2; viene descrito en un comentario dentro del clasificador
    # La variable tolerancia se refiere, permite que si se llega a ese valor marcado el clasificador para de iterar
    # La constante, es la inversa de la fuerza de regulacion
    # El sesgo, permite incentivar unas respuestas frente a otras, util para corregir con mayor profundidad

    verificador_rl = Verificados_RegresionLogistica_RL(penalizacion="l2",tolerancia=0.1,constante=1.0,sesgo=True,iteraciones_maximas=1000)

    diabetes_sin_simple = verificador_rl.clasificate(prepro=False,tipo_validacion=1,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/pima-indians-diabetes.data")
    diabetes_con_simple = verificador_rl.clasificate(prepro=True,tipo_validacion=1,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/pima-indians-diabetes.data")    
    diabetes_sin_cruzada = verificador_rl.clasificate(prepro=False,tipo_validacion=2,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/pima-indians-diabetes.data")
    diabetes_con_cruzada = verificador_rl.clasificate(prepro=True,tipo_validacion=2,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/pima-indians-diabetes.data")


    wdbc_sin_simple = verificador_rl.clasificate(prepro=False,tipo_validacion=1,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/wdbc.data")
    wdbc_con_simple = verificador_rl.clasificate(prepro=True,tipo_validacion=1,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/wdbc.data")
    wdbc_sin_cruzada = verificador_rl.clasificate(prepro=False,tipo_validacion=2,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/wdbc.data")
    wdbc_con_cruzada = verificador_rl.clasificate(prepro=True,tipo_validacion=2,porcentaje=0.75,folds=3,archivo="ConjuntosDatos/wdbc.data")

    resultados_sk_sin = [[round(diabetes_sin_simple, 3), round(wdbc_sin_simple, 3), 
        round(diabetes_con_simple,3), round(wdbc_con_simple,3)],  
        [round(diabetes_sin_cruzada, 3), round(wdbc_sin_cruzada, 3), 
        round(diabetes_con_cruzada,3), round(wdbc_con_cruzada,3)]]

    print("SKLearn:")
    print("Regresion Logistica")
    print(tabulate(resultados_sk_sin, headers=['Tasa de error', 'Diabetes(Sin Prepro)', 'Wdbc(Sin Prepro)','Diabetes(Con Prepro)','Wdbc(Con Prepro)'], showindex=['Val. Simple', 'Val. Cruzada'], tablefmt='fancy_grid'))

"""



