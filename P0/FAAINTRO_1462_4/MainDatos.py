from Datos import Datos
import sys

if __name__ == "__main__":
    fileName = "ConjuntosDatos/tic-tac-toe.data"
    datos = Datos(fileName)
    print(datos.nominalAtributos)
    print(datos.atributos)
    print(datos.datos)
    print(datos.diccionario)