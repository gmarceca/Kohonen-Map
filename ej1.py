import numpy as np
from mpl_toolkits import mplot3d
from random import shuffle

import matplotlib
matplotlib.use('TkAgg') #TEMP
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self, data, validatingDataSet, oja=True):
        # Seteamos una seed
        np.random.seed(1)

        self.trainingDataSet = data[:, 1:data.shape[1]]
        self.validatingDataSet = validatingDataSet[:, 1:data.shape[1]]

        # Obtenemos las categorias de la data
        self.trainingDataLabels = data[:, 0:1]
        self.validatingDataLabels = validatingDataSet[:, 0:1]

        # Definimos el numero de iteraciones que queremos correr
        self.numeroDeIteraciones = 100

        # Seteamos las dimesiones de los datos
        self.dimensionSalida = 3
        self.dimensionEntrada = self.trainingDataSet.shape[1]

        # Inicializo los pesos en valores random
        self.weights = np.array((self.dimensionSalida, 0), dtype=float)
        self.randoms = np.random.rand(self.dimensionSalida, self.dimensionEntrada);
        self.weights = (2 * self.randoms - 1) * 0.25

        self.learningValue = 0.01

        # Definimos si queremos correr la regla de oja o sanger
        self.oja = oja

    def train(self):
        count = 0
        while (count < self.numeroDeIteraciones) :
            for index in range(0, self.trainingDataSet.shape[0]) :
                x = self.trainingDataSet[index].reshape(self.dimensionEntrada, 1)

                # Calculamos las salidas S en forma vectorial S
                salidas = np.dot(self.weights, x)
                assert(salidas.shape == (self.dimensionSalida, 1))

                if (self.oja): # Regla de Oja en forma vectorial
                    self.weights = self.weights + self.learningValue * np.dot(salidas, (x - np.dot(self.weights.T, salidas)).T)

                else: # Regla de Sanger
                    for i in range(0, self.dimensionSalida):
                        for j in range(0, self.dimensionEntrada):
                            suma = 0
                            for k in range(i + 1):
                                suma += salidas[k, 0] * self.weights[k, j]
                            self.weights[i,j] =  self.weights[i,j] + self.learningValue * salidas[i, 0] * (x[j, 0] - suma)

            count += 1

        # Creamos los outpus vacios
        trainingOutputs = np.empty((self.trainingDataSet.shape[0], 3), dtype=float)
        validationgOutputs = np.empty((self.validatingDataSet.shape[0], 3), dtype=float)

        # Calculamos las salidas
        for i in range(self.trainingDataSet.shape[0]):
            trainingOutputs[i] = np.dot(self.trainingDataSet[i], np.transpose(self.weights))
        for i in range(self.validatingDataSet.shape[0]):
            validationgOutputs[i] = np.dot(self.validatingDataSet[i], np.transpose(self.weights))

        # Creamos el grafico
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        fig.suptitle('Regla Oja' if self.oja else 'Regla Sanger')

        # Definimos los colores
        colors = ['gray', 'black', 'red', 'orange', 'green', 'blue', 'magenta', 'purple', 'brown']

        # Graficamos las salidas
        for i in range(trainingOutputs.shape[0]):
            ax.scatter3D(trainingOutputs[i][0], trainingOutputs[i][1], trainingOutputs[i][2], c=colors[int(self.trainingDataLabels[i][0])-1],  marker='o');
        for i in range(validationgOutputs.shape[0]):
           ax.scatter3D(validationgOutputs[i][0], validationgOutputs[i][1], validationgOutputs[i][2], c=colors[int(self.validatingDataLabels[i][0])-1],  marker='x');

        plt.show()

if __name__ == '__main__':
    filename = 'tp2_training_dataset.csv'
    raw_data = open(filename, 'rt')
    data = np.loadtxt(raw_data, delimiter=",", usecols=range(851))

    # Dividimos la data en un set de training y un set de validacion
    trainingDataSet = data[0:850]
    validatingDataSet = data[850:900]

    nn = NeuralNetwork(trainingDataSet, validatingDataSet, True)
    nn.train()
