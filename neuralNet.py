import menu
import random
import math


class Neuron:
    """Neuron object for a neural network"""
    bias = -1  # Bias node static across all instances of neuron

    def __init__(self, number_of_inputs):
        """Constructor for a new neuron requires the number of inputs not including the bias node"""
        self.weight = []  # list to contain all the weights of the inputs
        for _ in range(0, number_of_inputs):
            random_weight = (random.random() * 2) - 1
            self.weight.append(random_weight)
        self.weight.append((random.random() * 2) - 1)  # To append one extra random weight for the bias node

    def excite_neuron(self, input_list):
        """Takes a list of inputs and tests weather the neuron becomes excited"""
        sum_excitement = 0
        for item in range(len(input_list)):
            sum_excitement += (input_list[item] * self.weight[item])
        sum_excitement += (self.bias * self.weight[-1])  # Don't forget to add the bias
        return 1/(1 + math.e**sum_excitement)


class NeuralNet:
    """Provides a structure to contain neurons in a neural network"""

    def __init__(self):
        """Constructor"""
        self.node_layer = []

    def add_node_layer(self, number_of_neurons, number_of_inputs=0):
        """Creates a layer of neuron nodes
                :param number_of_inputs: The number of inputs that will be passed into the network
                :param number_of_neurons: The number of neurons there will be in the network"""
        if len(self.node_layer) != 0:
            number_of_inputs = len(self.node_layer[-1])
        neuron_list = []
        for _ in range(0, number_of_neurons):
            neuron_list.append(Neuron(number_of_inputs))
        self.node_layer.append(neuron_list)

    def make_predictions(self, data):
        """Makes prediction based on input data passed"""
        if len(self.node_layer) != 0:
            for layer in range(len(self.node_layer)):
                prediction = []
                for row in range(len(data)):
                    excited_neurons = []
                    for node in range(len(self.node_layer[layer])):
                        neuron = self.node_layer[layer][node]
                        excited_neurons.append(neuron.excite_neuron(data[row]))
                    prediction.append(excited_neurons)
                data = prediction
        else:
            prediction = []
        return prediction


def pre_process_diabetes():
    """Pre processes the diabetes data and return pre processed data"""
    diabetes_data = []
    diabetes_target = []
    diabetes = menu.diabetes
    for row in range(len(diabetes)):
        for col in range(len(diabetes[row])):
            diabetes[row][col] = float(diabetes[row][col])
        diabetes_data.append(diabetes[row][:-1])
        diabetes_target.append(int(diabetes[row][-1]))
    trainingData, trainingTarget, testData, testTarget = menu.loadData(diabetes_data, diabetes_target)
    trainingData, testData = menu.normalize(trainingData, testData)
    return trainingData, trainingTarget, testData, testTarget


def pre_process_iris():
    """Pre process iris data and return pre processed data"""
    trainingData, trainingTarget, testData, testTarget = menu.loadData(menu.iris.data, menu.iris.target)
    trainingData, testData = menu.normalize(trainingData, testData)
    return trainingData, trainingTarget, testData, testTarget


def main():
    """Designed to show how the neuralNet class works"""
    print('Running iris data...')
    trainingData, trainingTarget, testData, testTarget = pre_process_iris()
    brain = NeuralNet()
    brain.add_node_layer(4, len(trainingData[0]))
    prediction = brain.make_predictions(trainingData)
    print('One layer neural net output')
    print(prediction)
    brain.add_node_layer(3)  # no need to pass second arg if this is not the first node layer
    prediction = brain.make_predictions(trainingData)
    print('Here is the output after adding another layer and running the same data')
    print(prediction)

    print('Running diabetes data...')
    trainingData, trainingTarget, testData, testTarget = pre_process_diabetes()
    diabetes_net = NeuralNet()
    diabetes_net.add_node_layer(6, len(trainingData[0]))
    diabetes_net.add_node_layer(7)
    diabetes_net.add_node_layer(3)
    diabetes_net.add_node_layer(1)
    prediction = diabetes_net.make_predictions(trainingData)
    print('This network has 3 hidden layers')
    print(prediction)


if __name__ == '__main__':
    main()
