#!/usr/bin/python

import pandas as pd
import numpy as np
import string

class Datos:

    nominalAtributos = None
    datos = None
    diccionario = None

    # TODO: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionario
    def __init__(self, nombreFichero):

        datosEntrada = pd.read_csv(nombreFichero)
        
        #============================= NOMINALATRIBUTOS ===================================== #

        # ARRAY de dimension (columnas) de tipo booleano
        nominalAtributos = np.empty([len(datosEntrada.columns)], dtype=bool)
        
        # Descartamos la ultima columna, que solo nos indica el valor de la clase
        for i in range(len(datosEntrada.columns)):
            aux = datosEntrada.values[0][i]

            # casteamos si es posible de string ==> float (incluimos reales y enteros)
            try:
                float(aux)
                nominalAtributos[i] = False
            except ValueError:
                nominalAtributos[i] = True    


        # =========================== DICCIONARIO ============================ #

        diccionario={}

        # revisamos si son valores nominales o enteros/reales
        for h in range(len(datosEntrada.columns)):
            if(nominalAtributos[h] == True):
                #Creamos la lista auxiliar, en la que aniadiremos los valores de una columna, sin repeticion
                lista_aux =[]
                for i in range(len(datosEntrada)):
                    if datosEntrada.values[i][h] not in lista_aux:
                        lista_aux.append(datosEntrada.values[i][h])
          
                #Lista con los elementos albergados y listo para ordenar
                lista_aux.sort()

                #Creamos diccionario auxiliar, donde insertar las claves correctamente junto alos valores correspondientes
                diccionario_aux = {}
                for j in range (len(lista_aux)):
                    diccionario_aux[lista_aux[j]] = j

                diccionario[datosEntrada.columns[h]] = diccionario_aux

                #Liberamos los contenidos de la lista y el diccionario auxiliar para la siguiente columna (atributo)
                diccionario_aux=None
                lista_aux=None
            else:
                diccionario[datosEntrada.columns[h]] = {}


        # ============================= DATOS =============================== #

        # ARRAY bidimensional de tipo string, usamos numero de filas y de columnas de la variable datosEntrada
        datos = np.empty([len(datosEntrada),len(datosEntrada.columns)],dtype=object)
        

        for i in range(len(datosEntrada)): #filas
            for j in range(len(datosEntrada.columns)): #columnas

                # Comprobamos si el tipo de dato es nominal
                if ( nominalAtributos[j] == True ):
                    # Buscamos en el diccionario sus pares clave-valor
                    valor = diccionario[datosEntrada.columns[j]]

                    # Encontramos la conversion del atributo nominal a numerico
                    entrada = valor[datosEntrada.values[i][j]]

                #Si el dato ya es de tipo numerico, se introduce
                else:
                    entrada = datosEntrada.values[i][j]
                datos[i][j] = entrada
        
    # TODO: implementar en la practica 1
    def extraeDatos(self, idx):
        pass