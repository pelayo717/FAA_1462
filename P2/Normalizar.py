import numpy as np

class Normalizar():
    
    medias = None
    desviaciones = None
    def __init__(self):

        self.medias = []
        self.desviaciones = []

    def calcularMediasDesv(self,datos,nominalAtributos):

        for i in range(len(nominalAtributos)):
            if(nominalAtributos[i] == False): # Enteros o decimales
                if(isinstance(datos[0][i],float) == True):
                    aux_numpy = np.zeros(len(datos),dtype=float)
                    for j in range(len(datos)):
                        aux_numpy[j] = datos[j][i]
                    
                    self.medias.append(np.mean(aux_numpy))
                    self.desviaciones.append(np.std(aux_numpy))
   
    def  normalizarDatos(self,datos,nominalAtributos):
        conteo_lista_decimales = 0
        for i in range(len(nominalAtributos)):
            if(nominalAtributos[i] == False): # Enteros o decimales
                if(isinstance(datos[0][i],float) == True):
                    for j in range(len(datos)):
                        datos[j][i] = (datos[j][i] - self.medias[conteo_lista_decimales])/self.desviaciones[conteo_lista_decimales]
                    conteo_lista_decimales+=1