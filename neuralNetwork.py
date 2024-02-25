import numpy
from scipy.special import expit


class neuralNetwork:
    def __init__(self, input_nodes, output_nodes, count_hidden_layers, count_hidden_nodes_per_layer, learning_rate):
        #Количество узлов входного, скрытого и выходного слоя
        self.inputNodes = input_nodes
        self.hiddenLayers = count_hidden_layers
        self.hiddenNodesPerLayer = count_hidden_nodes_per_layer
        self.outputNodes = output_nodes

        #Коэффициент обучения
        self.learningRate = learning_rate

        #Матрицы весовых коэффициентов
        self.matrixs = self.getMatrixsHiddenLayers(self.inputNodes, self.outputNodes, self.hiddenLayers, self.hiddenNodesPerLayer)

        #Функция активации
        self.activationFunction = lambda x: expit(x)

    def getMatrixsHiddenLayers(self, count_input_nodes, count_output_nodes, hidden_layers_count, hidden_nodes_per_layer):
        #Инициализация весовых коэффициентов для входного слоя нейронной сети
        matrixs = [self.getMatrixCoefficients(count_input_nodes, hidden_nodes_per_layer)]

        #Инициализация весовых коэффициентов для скрытых слоёв нейронной сети
        for _ in range(0, hidden_layers_count):
            matrixs.append(self.getMatrixCoefficients(hidden_nodes_per_layer, hidden_nodes_per_layer))

        #Инициализация весовых коэффициентов для выходного слоя нейронной сети
        matrixs.append(self.getMatrixCoefficients(hidden_nodes_per_layer, count_output_nodes))
        return matrixs

    def getMatrixCoefficients(self, input_count_nodes, output_count_nodes):
        #Инициализация весовых коэффициентов слоя нейронной сети нормальным распределением
        return numpy.random.normal(0.0, pow(input_count_nodes, -0.5), (output_count_nodes, input_count_nodes))

    def train(self, inputs_list, targets_list):
        #Преобразование входных одномерных массивов в транспонированные матрицы
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        #Процесс обучения нейронной сети
        signals = [inputs]
        for layer in self.matrixs:
            signals.append(self.get_signals(layer, signals[-1]))

        #Обратное распространение информации об ошибках по слоям нейронной сети
        output_errors = targets - signals[-1]
        for i in range(len(self.matrixs) - 1, -1, -1):
            self.matrixs[i] += self.learningRate * numpy.dot((output_errors * signals[i + 1] * (1.0 - signals[i + 1])), numpy.transpose(signals[i]))
            output_errors = numpy.dot(self.matrixs[i].T, output_errors)

    def query(self, inputs_list):
        #Преобразование входных значений в двумерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        #Расчёт сигналов нейронной сети
        for layer in self.matrixs:
            inputs = self.get_signals(layer, inputs)

        return inputs

    def get_signals(self, weight_coefficients, network_layer):
        #Расчёт сигналов слоя нейронной сети
        return self.activationFunction(numpy.dot(weight_coefficients, network_layer))
