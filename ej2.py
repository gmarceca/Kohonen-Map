import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import math as math
from random import shuffle
from sklearn.decomposition import PCA

class SOM:
    def __init__(self, data, validatingDataSet, trainingLabels, validatingLabels):
        # Seteamos una seed
        np.random.seed(1)

        self.data = data
        self.validatingDataSet = validatingDataSet
        self.trainingDataLabels = trainingLabels
        self.validatingDataLabels = validatingLabels

        # Definimos el numero de iteraciones que queremos correr
        self.numeroDeIteraciones = 1000

        # Seteamos las dimesiones de los datos
        self.dimensionSalida = 2
        self.dimensionEntrada = self.data.shape[1]

        # Seteamos el tamaño del plano
        self.sizeX = 25
        self.sizeY = 25

        # Seteamos los pesos de manera random
        self.weights = np.array((self.sizeX , self.sizeY, self.dimensionEntrada), dtype=float)
        self.randoms = np.random.rand(self.sizeX, self.sizeY, self.dimensionEntrada);
        self.weights = self.randoms

        # Seteamos los parametros iniciales para el enfriamento
        self.radioInicial = max(self.sizeX, self.sizeY) / 2
        self.t1 = 1000 / math.log10(self.radioInicial)
        self.t2 = 1000
        self.initLearningRate = 0.1

    def train(self):
        count = 0
        while (count < self.numeroDeIteraciones) :
            # Elegimos un dato random
            randomEntrie = np.random.randint(low=0, high=self.data.shape[0])
            inputX = self.data[randomEntrie].reshape(self.dimensionEntrada,1)

            # Buscamos el indice que mas se parece a x
            indexMinimo = self.encontrarMinimo(inputX)

            # Obtenemos el radio y el learning rate para esta iteracion
            radio = self.obtenerRadio(count)
            learningRate = self.obtenerLearningRate(count)

            for x in range(self.weights.shape[0]):
                for y in range(self.weights.shape[1]):
                    # Para cada peso calculamos la distancia entre el peso y el indice minimo
                    weight = self.weights[x][y].reshape(inputX.shape[0], 1)
                    distancia = np.sum((np.array([x, y]) - indexMinimo) ** 2)
                    # Si esta en la zona de vecinidad actualizamos los pesos
                    if distancia <= radio ** 2:
                        vecinidad = self.obtenerVecinidad(distancia, radio)
                        self.weights[x][y] = (weight + (learningRate * vecinidad * (inputX - weight))).reshape(inputX.shape[0])
            count += 1

        # Creamos los outpus vacios
        trainingOutputs = np.empty((self.data.shape[0], 2), dtype=float)
        validationgOutputs = np.empty((validatingDataSet.shape[0], 2), dtype=float)

        # Definimos los colores. Los de entramiento tienen un alpha asi se nota la densidad de inputs que caen en un cuadrante
        colors = ['#D3D3D395', '#00000095', '#FF000095', '#FFA50095', '#00800095', '#0000FF95', '#FF00FF95', '#80008095', '#A52A2A95']
        validatingColors = ['#D3D3D3', '#000000', '#FF0000', '#FFA500', '#008000', '#0000FF', '#FF00FF', '#800080', '#A52A2A']

        # Creamos el grafico
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.suptitle('Gráfico con input de ' + str(self.dimensionEntrada) + ' dimensiones')

        # Calculamos los outputs y los graficamos
        for i in range(trainingOutputs.shape[0]):
            trainingOutputs[i] = self.encontrarMinimo(self.data[i].reshape(self.dimensionEntrada, 1))
            plt.plot(trainingOutputs[i][0]+.5, trainingOutputs[i][1]+.5,  marker='o', color=colors[int(self.trainingDataLabels[i][0])-1])

        for i in range(validatingDataSet.shape[0]):
            validationgOutputs[i] = self.encontrarMinimo(self.validatingDataSet[i].reshape(self.dimensionEntrada, 1))
            plt.plot(validationgOutputs[i][0]+.75, validationgOutputs[i][1]+.75 ,  marker='X', color=validatingColors[int(self.validatingDataLabels[i][0])-1])

        # Terminamos de configurar el grafico
        ax.set_xlim([0, self.sizeX])
        ax.set_ylim([0, self.sizeY])
        plt.xticks(np.arange(self.sizeX))
        plt.yticks(np.arange(self.sizeY))
        ax.grid(which='both')
        plt.show()

    # Esta funcion encuentra el peso mas cercano al input dado. Devuelve los indices de ese peso
    def encontrarMinimo(self, inputX):
        indiceMinimo = np.array([0, 0])
        # Calculamos la primera distancia
        primeraDistancia = np.sum((self.weights[0][0].reshape(self.dimensionEntrada, 1) - inputX) ** 2)
        distanciaMinima = primeraDistancia

        for x in range(self.weights.shape[0]):
            for y in range(self.weights.shape[1]):
                # Para todo peso vemos si la distancia es menor a la que tengo guardada
                weight = self.weights[x][y].reshape(self.dimensionEntrada, 1)
                distancia = np.sum((weight - inputX) ** 2)
                if distancia < distanciaMinima:
                    distanciaMinima = distancia
                    indiceMinimo = np.array([x, y])
        return indiceMinimo

    # Obtiene el tamaño del radio en la iteracion i
    def obtenerRadio(self, i):
        return self.radioInicial * np.exp(-i / self.t1)

    # Obtiene el valor del learing rate en la iteracion i
    def obtenerLearningRate(self, i):
        learningRate = self.initLearningRate * np.exp(-i / self.t2)
        return learningRate

    # Obtiene la funcion de vecinidad
    def obtenerVecinidad(self, distancia, varianza):
        return np.exp(- distancia / (2 * (varianza ** 2)))

if __name__ == '__main__':
    filename = 'tp2_training_dataset.csv'
    raw_data = open(filename, 'rt')
    data = np.loadtxt(raw_data, delimiter=",", usecols=range(851))

    originalData = True
    threeDimesionData = False
    nineDimesionData = False

    # Obtenemos las categorias de la data
    trainingLabels = data[0:850, 0:1]
    validatingLabels = data[850:900, 0:1]

    if originalData == True:
        trainingDataSet = data[0:850, 1:851]
        validatingDataSet = data[850:900, 1:851]

    elif threeDimesionData == True:
        # Transformamos la data en 3 dimeniones
        pca = PCA(n_components = 3)
        principalComponents = pca.fit_transform(data[:, 1:851])

        trainingDataSet = principalComponents[0:850]
        validatingDataSet = principalComponents[850:900]

    elif nineDimesionData == True:
        # Transformamos la data en 9 dimeniones
        pca = PCA(n_components = 9)
        principalComponents = pca.fit_transform(data[:, 1:851])

        trainingDataSet = principalComponents[0:850]
        validatingDataSet = principalComponents[850:900]

    som = SOM(trainingDataSet, validatingDataSet, trainingLabels, validatingLabels)
    som.train()
