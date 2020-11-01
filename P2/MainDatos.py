from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from Clasificador import *
from Distancias import *
from Verificador import *
from MatrizConfusion import MatrizConfusion
from tabulate import tabulate

if __name__ == "__main__":

    fileName = "ConjuntosDatos/pima-indians-diabetes.data"
    datos_diabetes = Datos(fileName)

    fileName = "ConjuntosDatos/wdbc.data"
    datos_wdbc = Datos(fileName)

    knn = ClasificadorVecinosProximos(3, Euclidea)
    cl = ClasficadorRegresionLogistica(0.5,20)

    ############################## Diabetes ##############################
    validacion_simple_diabetes = ValidacionSimple(75,10)
    simple_diabetes = validacion_simple_diabetes.creaParticiones(datos_diabetes)

    validacion_cruzada_diabetes = ValidacionCruzada(6)
    cruzada_diabetes = validacion_cruzada_diabetes.creaParticiones(datos_diabetes)
    
    medias_simples_diabetes_rl = cl.validacion(validacion_simple_diabetes, datos_diabetes)
    medias_cruzadas_diabetes_rl = cl.validacion(validacion_cruzada_diabetes, datos_diabetes)

    medias_simples_diabetes = knn.validacion(validacion_simple_diabetes, datos_diabetes)
    medias_cruzadas_diabetes = knn.validacion(validacion_cruzada_diabetes, datos_diabetes)
    
    

    ############################## Wdbc ##############################
    validacion_simple_wdbc = ValidacionSimple(75,10)
    simple_wdbc = validacion_simple_wdbc.creaParticiones(datos_wdbc)

    validacion_cruzada_wdbc = ValidacionCruzada(6)
    cruzada_wdbc = validacion_cruzada_wdbc.creaParticiones(datos_wdbc)

    medias_simples_wdbc_rl = cl.validacion(validacion_simple_wdbc, datos_wdbc)
    medias_cruzadas_wdbc_rl = cl.validacion(validacion_cruzada_wdbc, datos_wdbc)
    
    medias_simples_wdbc = knn.validacion(validacion_simple_wdbc, datos_wdbc)
    medias_cruzadas_wdbc = knn.validacion(validacion_cruzada_wdbc, datos_wdbc)


    # Guardamos resultados de los resultados KNN
    resultados_P2 = [[round(medias_simples_diabetes[0], 3), round(medias_simples_wdbc[0], 3)], 
            [round(medias_cruzadas_diabetes[0], 3), round(medias_cruzadas_wdbc[0], 3)]]

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
    print("\nDiabetes KNN")
    tpr, fpr = mx1.matrix_media(medias_simples_diabetes[1], 
        medias_cruzadas_diabetes[1], 
        medias_simples_diabetes[2], 
        medias_cruzadas_diabetes[2],
        medias_simples_diabetes[3], 
        medias_cruzadas_diabetes[3],
        medias_simples_diabetes[4], 
        medias_cruzadas_diabetes[4])

    plot_points = [[fpr, tpr, 'KNN']]
    mx1.plot(plot_points, "Diabetes")

    print("\nDiabetes Regresion Logistica")
    tpr, fpr = mx1.matrix_media(medias_simples_diabetes_rl[1], 
        medias_cruzadas_diabetes_rl[1], 
        medias_simples_diabetes_rl[2], 
        medias_cruzadas_diabetes_rl[2],
        medias_simples_diabetes_rl[3], 
        medias_cruzadas_diabetes_rl[3],
        medias_simples_diabetes_rl[4], 
        medias_cruzadas_diabetes_rl[4])

    plot_points = [[fpr, tpr, 'Reg. Log.']]
    mx1.plot(plot_points, "Diabetes")

    # WDBC
    print("\nWdbc KNN")
    tpr, fpr = mx1.matrix_media(medias_simples_wdbc[1], 
        medias_cruzadas_wdbc[1], 
        medias_simples_wdbc[2], 
        medias_cruzadas_wdbc[2],
        medias_simples_wdbc[3], 
        medias_cruzadas_wdbc[3],
        medias_simples_wdbc[4], 
        medias_cruzadas_wdbc[4])
    
    plot_points = [[fpr, tpr, 'KNN']]
    mx1.plot(plot_points, "Wdbc")

    print("\nWdbc Regresion Logistica")
    tpr, fpr = mx1.matrix_media(medias_simples_wdbc_rl[1], 
        medias_cruzadas_wdbc_rl[1], 
        medias_simples_wdbc_rl[2], 
        medias_cruzadas_wdbc_rl[2],
        medias_simples_wdbc_rl[3], 
        medias_cruzadas_wdbc_rl[3],
        medias_simples_wdbc_rl[4], 
        medias_cruzadas_wdbc_rl[4])
    
    plot_points = [[fpr, tpr, 'Reg. Log.']]
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
