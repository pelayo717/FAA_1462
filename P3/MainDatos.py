from Datos import Datos
from EstrategiaParticionado import *
from Clasificador import *
from Distancias import *
from Verificador import *
from MatrizConfusion import MatrizConfusion
from tabulate import tabulate
import sys

if __name__ == "__main__":

    # Abrimos los ficheros y extraemos los datos
    if(len(sys.argv) < 2):
        print("Fallo en dataset a emplear")
        exit()
    fileName = sys.argv[1]
    fileName2="ConjuntosDatos/tic-tac-toe.data"
    datos_tictactoe = Datos(fileName2)

    # Creamos las validaciones
    validacion_simple = ValidacionSimple(75,10)
    validacion_cruzada = ValidacionCruzada(6)

    ag_tictactoe = ClasficadorAlgoritmoGenetico( 
                        tam_poblacion=150, cond_terminacion=150, max_reglas=10, 
                        tipo_cruce = 1, tipo_mutacion=1, prob_cruce=0.5, 
                        prob_mutacion=0.5)

    medias_tictactoe = ag_tictactoe.validacion(validacion_simple, datos_tictactoe, filename=fileName2)
    mx1_tictactoe = MatrizConfusion()

    print("\nAlgoritmo Genetico: " + str(medias_tictactoe[0]))
    ag_tpr_tictactoe, ag_fpr_tictactoe = mx1_tictactoe.matrix_media(medias_tictactoe[1], 
                                medias_tictactoe[1], 
                                medias_tictactoe[2], 
                                medias_tictactoe[2], 
                                medias_tictactoe[3], 
                                medias_tictactoe[3], 
                                medias_tictactoe[4], 
                                medias_tictactoe[4])

        # Abrimos un fichero de resultados para albergar y automatizar el proceso
    """fp = open("resultados_poblacion.txt","wb")
    fp.write("Pobla|Term|Max|Cruce|Mutacion|ProbCruce|ProbMutaci|AlgGenetico")

    aux=10
    while(aux <= 100):

        ag = ClasficadorAlgoritmoGenetico( 
                            tam_poblacion=aux, cond_terminacion=150, max_reglas=25, 
                            tipo_cruce = 0, tipo_mutacion=0, prob_cruce=0.25, 
                            prob_mutacion=0.05)

        medias = ag.validacion(validacion_simple, datos_titanic, filename="ConjuntosDatos/titanic.data")

        fp.write(str(aux)+ "|150|25|0|0|0.25|0.05|" + str(medias[0]))
        print
        aux+=10
        #mx1 = MatrizConfusion()
        #print("\nAlgoritmo Genetico: " + str(medias[0]))
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
    fp.close()"""