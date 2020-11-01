import numpy as np

class Normalizar():
    
    medias = None
    desviaciones = None

    def __init__(self):

        self.medias = []
        self.desviaciones = []


    def calcularMediasDesv(self,datos,nominalAtributos):
        # Miramos todos los atributos de un dataset, para conocer que tipo de atributo es cada uno (Enteros o decimales) o Nominales
        for i in range(len(nominalAtributos)):
            if(nominalAtributos[i] == False): # Caso en el que son enteros o decimales
                if(isinstance(datos[0][i],float) == True): # Comprobamos que sean decimales
                    aux_numpy = np.zeros(len(datos),dtype=float) # Creamos un array para albergar todos los valores de la columna concreta
                    for j in range(len(datos)):
                        aux_numpy[j] = datos[j][i]
                    
                    self.medias.append(np.mean(aux_numpy)) # Sacamos la media de toda la columna 
                    self.desviaciones.append(np.std(aux_numpy)) # Sacamos la desviacion 
   
    def  normalizarDatos(self,datos,nominalAtributos):
        conteo_lista_decimales = 0
        for i in range(len(nominalAtributos)):
            if(nominalAtributos[i] == False): # Caso en el que son enteros o decimales
                if(isinstance(datos[0][i],float) == True): # Comprobamos que sean decimales
                    for j in range(len(datos)): # Para cada valor de la columna, calculamos su nuevo valor correspondiente y se le asigna 
                        datos[j][i] = (datos[j][i] - self.medias[conteo_lista_decimales])/self.desviaciones[conteo_lista_decimales]
                    conteo_lista_decimales+=1


