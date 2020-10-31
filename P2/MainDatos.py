from Datos import Datos
from EstrategiaParticionado import ValidacionSimple
from Clasificador import *
from Distancias import *
from MatrizConfusion import MatrizConfusion
from tabulate import tabulate

if __name__ == "__main__":

    fileName = "ConjuntosDatos/pima-indians-diabetes.data"
    datos_diabetes = Datos(fileName)

    fileName = "ConjuntosDatos/wdbc.data"
    datos_wdbc = Datos(fileName)

    knn = ClasificadorVecinosProximos(3, Mahalanobis)
    cl = ClasficadorRegresionLogistica(0.0000001,10)

    ############################## Diabetes ##############################
    validacion_simple_diabetes = ValidacionSimple(75,10)
    simple_diabetes = validacion_simple_diabetes.creaParticiones(datos_diabetes)

    validacion_cruzada_diabetes = ValidacionCruzada(6)
    cruzada_diabetes = validacion_cruzada_diabetes.creaParticiones(datos_diabetes)
    
    medias_simples_diabetes = knn.validacion(validacion_simple_diabetes, datos_diabetes)
    medias_cruzadas_diabetes = knn.validacion(validacion_cruzada_diabetes, datos_diabetes)
    
    #medias_simples_diabetes_rl = cl.validacion(validacion_simple_diabetes, datos_diabetes)
    #medias_cruzadas_diabetes_rl = cl.validacion(validacion_cruzada_diabetes, datos_diabetes)

    ############################## Wdbc ##############################
    validacion_simple_wdbc = ValidacionSimple(75,10)
    simple_wdbc = validacion_simple_wdbc.creaParticiones(datos_wdbc)

    validacion_cruzada_wdbc = ValidacionCruzada(6)
    cruzada_wdbc = validacion_cruzada_wdbc.creaParticiones(datos_wdbc)
    
    medias_simples_wdbc = knn.validacion(validacion_simple_wdbc, datos_wdbc)
    medias_cruzadas_wdbc = knn.validacion(validacion_cruzada_wdbc, datos_wdbc)


    # Guardamos resultados de los resultados
    resultados_P2 = [[round(medias_simples_diabetes[0], 3), round(medias_simples_wdbc[0], 3)], 
            [round(medias_cruzadas_diabetes[0], 3), round(medias_cruzadas_wdbc[0], 3)]]

    # Impresion de las tablas
    print("Practica 2:")
    print(tabulate(resultados_P2, headers=['Tasa de error', 'Diabetes', 'Wdbc'], 
        showindex=['Val. Simple', 'Val. Cruzada'], tablefmt='fancy_grid'))



    ############################## MATRIZ CONFUSION ###########################################

    # Calculamos la media de las tasas de val. simple y val. cruzada para la matriz de confusion media
    mx1 = MatrizConfusion()

    # TIC-TAC-TOE
    print("\nDiabetes")
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

    # GERMAN
    print("\nWdbc")
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