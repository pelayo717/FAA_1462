from Datos import Datos
from EstrategiaParticionado import *
from Clasificador import *
from Distancias import *
from Verificador import *
from MatrizConfusion import MatrizConfusion
from tabulate import tabulate

if __name__ == "__main__":

    # Abrimos los ficheros y extraemos los datos
    fileName = "ConjuntosDatos/tic-tac-toe.data"
    datos_titanic = Datos(fileName)


    # Creamos las validaciones
    validacion_simple = ValidacionSimple(75,10)

    validacion_cruzada = ValidacionCruzada(6)
        
    ag = ClasficadorAlgoritmoGenetico( 
                        tam_poblacion=100, cond_terminacion=150, max_reglas=35, 
                        tipo_cruce = 0, tipo_mutacion=0, prob_cruce=0.3, 
                        prob_mutacion=0.25)

    medias = ag.validacion(validacion_simple, datos_titanic, filename=fileName)
    
    del ag

    fp = open("resultados_poblacion2.txt","a")
    fp.write("100|150|35|0|0|0.3|0.25|" + str(medias[0])+"|\n")
    fp.close()

    mx1 = MatrizConfusion()

    print("\nAlgoritmo Genetico: " + str(medias[0]))
    tpr, fpr = mx1.matrix_media(medias[1], 
        medias[1], 
        medias[2], 
        medias[2],
        medias[3], 
        medias[3],
        medias[4], 
        medias[4])
    
    plot_points = [[fpr, tpr, 'AG']]
    mx1.plot(plot_points, "Titanic")


