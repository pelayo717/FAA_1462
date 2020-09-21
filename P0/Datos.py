#!/usr/bin/python

import pandas as pd
import numpy as np
import string

class Datos:

    nominalAtributos = None
    datos = None

    # TODO: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionario
    def __init__(self, nombreFichero):

        datosEntrada = pd.read_csv(nombreFichero)
        
        #============================= NOMINALATRIBUTOS =====================================#

        # ARRAY de dimension (columnas) de tipo booleano
        nominalAtributos = np.empty([len(datosEntrada.columns)-1], dtype=bool)
        
        # Descartamos la ultima columna, que solo nos indica el valor de la clase
        for i in range(len(datosEntrada.columns) - 1):
            aux = datosEntrada.values[:1][0][i]

            # casteamos si es posible de string ==> float (incluimos reales y enteros)
            try:
                float(aux)
                nominalAtributos[i] = False
            except ValueError:
                nominalAtributos[i] = True
            
        # Verificamos que se han guardado correctamente los booleanos de cada columna
        print(nominalAtributos)

        # ============================= DATOS ===============================#

        # ARRAY bidimensional de tipo string, usamos numero de filas y de columnas de la variable datosEntrada
        datos = np.empty([len(datosEntrada),len(datosEntrada.columns)-1],dtype=object)
        
        for i in range(len(datosEntrada)): #filas
            for j in range(len(datosEntrada.columns)-1): #columnas
                datos[i][j]=str(datosEntrada.values[:][i][j])


        # Verificamos que se han guardado correctamente los objetos de cada fila
        print(datos)

        # =========================== DICCIONARIO ============================#

    # TODO: implementar en la practica 1
    def extraeDatos(self, idx):
        pass