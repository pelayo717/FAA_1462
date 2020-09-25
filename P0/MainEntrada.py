from Datos import Datos
import sys

if __name__ == "__main__":

    if ( len(sys.argv) != 2):
        print("Numero de argumentos de entrada incorrectos.")

    else:
        fileName = "ConjuntosDatos/"  + sys.argv[1]
        datos = Datos(fileName)
