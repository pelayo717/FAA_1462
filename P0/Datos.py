#!/usr/bin/python

import pandas as pd
import numpy as np
import string

class Datos:

    nominalAtributos = None     # Array de tipo booleano que indica si el tipo de atributo es nominal
    atributos = None            # Array de tipo string con los nombres de los atributos
    datos = None                # Array bidimensional 'pandas' con los datos de los atributos
    diccionario = None          # Diccionario con los pares clave-valor de las conversiones de atributos nominales

    # TODO: procesar el fichero para asignar correctamente las variables nominalAtributos, datos y diccionario
    def __init__(self, nombreFichero):

        datosEntrada = pd.read_csv(nombreFichero)

        #=============================== ATRIBUTOS ======================================= #

        atributos = list(datosEntrada.columns)

        cantidadAtributos = len(atributos) # Variable auxiliar que almacena el numero de atributos del dataset
        cantidadDatos = len(datosEntrada)  # Variable auxiliar que almacena el numero de datos por columna del dataset
        
        #============================= NOMINALATRIBUTOS ===================================== #

        # ARRAY de dimension (columnas) de tipo booleano
        nominalAtributos = np.empty(cantidadAtributos, dtype=bool)
        
        # Descartamos la ultima columna, que solo nos indica el valor de la clase
        for i in range(cantidadAtributos):
            aux = datosEntrada.values[0][i]

            # casteamos si es posible de string ==> float (incluimos reales y enteros)
            try:
                float(aux)
                nominalAtributos[i] = False
            except ValueError:
                nominalAtributos[i] = True    


        # =========================== DICCIONARIO ============================ #

        diccionario={}

        # Revisamos si son valores nominales o enteros/reales
        for h in range(cantidadAtributos):
            if(nominalAtributos[h] == True):
                # Creamos la lista auxiliar, en la que aniadiremos los valores de una columna, sin repeticion
                lista_aux =[]

                for i in datosEntrada[atributos[h]]:
                    if i not in lista_aux:
                        lista_aux.append(i)
          
                # Lista con los elementos albergados y listo para ordenar
                lista_aux.sort()

                #Creamos diccionario auxiliar, donde insertar las claves correctamente junto a los valores correspondientes
                diccionario_aux = {}
                for i in lista_aux:
                    diccionario_aux[i] = lista_aux.index(i)

                diccionario[atributos[h]] = diccionario_aux

                #Liberamos los contenidos de la lista y el diccionario auxiliar para la siguiente columna (atributo)
                diccionario_aux=None
                lista_aux=None
            else:
                diccionario[atributos[h]] = {}


        # ============================= DATOS =============================== #

        # ARRAY bidimensional de tipo numerico, usamos numero de filas y de columnas de la variable datosEntrada
        datos = np.empty([cantidadDatos, 0],dtype=object)
        
        for i in range(cantidadAtributos): # Para cada uno de los atributos
            nombreAtributo = atributos[i]
            columnaAtributos = []

            # Comprobamos si el tipo de dato es nominal
            if ( nominalAtributos[i] == True ):
                for j in datosEntrada[nombreAtributo]: # Para cada uno de los valores

                    # Buscamos en el diccionario sus pares clave-valor
                    valor = diccionario[nombreAtributo]

                    # Encontramos la conversion del atributo nominal a numerico
                    entrada = valor[j]
                    columnaAtributos.append(entrada)

            #Si el dato ya es de tipo numerico, se introduce
            else:
                for j in datosEntrada[nombreAtributo]: # Para cada uno de los valores
                    
                    columnaAtributos.append(j)
            
            columnaAtributos = np.array([columnaAtributos]).transpose()
            datos = np.append(datos, columnaAtributos, axis=1)

        
    # TODO: implementar en la practica 1
    def extraeDatos(self, idx):
        pass