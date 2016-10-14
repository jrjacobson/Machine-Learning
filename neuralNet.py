import menu
import sys
import random


class Neuron:
    """Neuron object for a neural network"""

    bias = -1

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
        if sum_excitement >= 0:
            is_excited = 1
        else:
            is_excited = 0
        return is_excited


class NeuralNet:
    """Provides a structure to contain neurons in a neural network"""

    def __init__(self):
        """Constructor"""
        self.node_layer = []

    def add_node_layer(self, number_of_neurons, number_of_inputs):
        """Creates a layer of neuron nodes
                :param number_of_inputs: The number of inputs that will be passed into the network
                :param number_of_neurons: The number of neurons there will be in the network"""
        neuron_list = []
        for _ in range(0, number_of_neurons):
            neuron_list.append(Neuron(number_of_inputs))
        self.node_layer.append(neuron_list)

    def make_predictions(self, data):
        """Makes prediction based on input data passed"""
        prediction = []
        for row in range(len(data)):
            excited_neurons = []
            for node in range(len(self.node_layer[0])):  # Logic will need to be changed here to deal with multiple layers in the neural net
                neuron = self.node_layer[0][node]
                excited_neurons.append(neuron.excite_neuron(data[row]))
            prediction.append(excited_neurons)
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


def main(args):
    """Designed to show how the neuralNet class works"""
    print('Running iris data...')
    trainingData, trainingTarget, testData, testTarget = pre_process_iris()
    brain = NeuralNet()
    brain.add_node_layer(3, len(trainingData[0]))
    prediction = brain.make_predictions(trainingData)
    print(prediction)

    print('Running diabetes data...')
    trainingData, trainingTarget, testData, testTarget = pre_process_diabetes()
    diabetes_net = NeuralNet()
    diabetes_net.add_node_layer(2, len(trainingData[0]))
    prediction = diabetes_net.make_predictions(trainingData)
    print(prediction)


if __name__ == '__main__':
    main(sys.argv)
