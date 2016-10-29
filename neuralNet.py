import menu
import random
import math


class Neuron:
    """Neuron object for a neural network"""
    bias = -1  # Bias node static across all instances of neuron

    def __init__(self, number_of_inputs):
        """Constructor for a new neuron requires the number of inputs not including the bias node"""
        self.weight = []  # list to contain all the weights of the inputs
        self.error = 0.0
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
        return 1 / (1 + math.e ** sum_excitement)


class NeuralNet:
    """Provides a structure to contain neurons in a neural network"""

    def __init__(self):
        """Constructor"""
        self.node_layer = []

    def back_prop(self, data, target):
        for epoch in range(300):
            # Run dataset
            for row in range(len(data)):
                # Run one row
                prediction = []
                row_data = data[row]  # this is so I can write over my data ever layer
                for layer in range(len(self.node_layer)):
                    # feed forward row
                    excited_neurons = []
                    for node in range(len(self.node_layer[layer])):
                        neuron = self.node_layer[layer][node]
                        excited_neurons.append(neuron.excite_neuron(row_data))
                    row_data = excited_neurons
                    prediction.append(excited_neurons)

                node_layer = -1
                for _ in range(len(self.node_layer)):
                    # Set all errors
                    nodes_in_layer = len(self.node_layer[node_layer])
                    for neuron_index in range(nodes_in_layer):
                        # Setting errors for each node in the layer
                        if node_layer == -1:
                            # last layer

                            # set the prediction to the highest values column number the set output to a one or zero
                            high = 0
                            predict = -1
                            for col in range(len(prediction[-1])):
                                if prediction[-1][col] > high:
                                    high = prediction[-1][col]
                                    predict = col
                            # is the prediction right or wrong
                            output = 0
                            if target[row] == predict:
                                output = 1

                            node_output = prediction[node_layer][neuron_index]
                            new_node_error = node_output * (1 - node_output) * (node_output - output)
                            self.node_layer[node_layer][neuron_index].error = new_node_error

                        else:
                            # hidden layer

                            sum_error = 0.0
                            num_of_next_layer_nodes = len(self.node_layer[node_layer + 1])
                            for next_layer_node in range(num_of_next_layer_nodes):
                                # this sums up the weight * error of all nodes in the layer to the right
                                node_error = self.node_layer[node_layer + 1][next_layer_node].error
                                node_weight = self.node_layer[node_layer + 1][next_layer_node].weight[next_layer_node]
                                sum_error += (node_error * node_weight)

                            # Calculate and set this nodes error
                            node_output = prediction[node_layer][neuron_index]
                            new_node_error = node_output * (1 - node_output) * sum_error
                            self.node_layer[node_layer][neuron_index].error = new_node_error
                    node_layer -= 1

                for current_layer in range(len(self.node_layer)):
                    # update weights
                    learning_rate = .1
                    for neurons in range(len(self.node_layer[current_layer])):
                        num_of_weights = len(self.node_layer[current_layer][neurons].weight)
                        for weight in range(num_of_weights):
                            bias = self.node_layer[current_layer][neurons].bias
                            if current_layer == 0:
                                if weight != (num_of_weights - 1):
                                    this_input = data[current_layer][weight]
                                else:
                                    this_input = bias
                            else:
                                if weight != (num_of_weights - 1):
                                    this_input = prediction[current_layer - 1][weight]
                                else:
                                    this_input = bias

                            old_weight = self.node_layer[current_layer][neurons].weight[weight]
                            error = self.node_layer[current_layer][neurons].error
                            new_weight = old_weight - learning_rate * error * this_input
                            self.node_layer[current_layer][neurons].weight[weight] = new_weight
            prediction = self.feed_forward(data)
            predictions = self.classify_predictions(prediction)
            percent = menu.test(target, predictions)
            print("%i%%" % percent)

    def add_node_layer(self, number_of_neurons, number_of_inputs=0):
        """Creates a layer of neuron nodes
                :param number_of_inputs: The number of inputs that will be passed into the network this will be
                overridden if the input is coming from a hidden layer
                :param number_of_neurons: The number of neurons there will be in the network"""
        if len(self.node_layer) != 0:
            number_of_inputs = len(self.node_layer[-1])
        neuron_list = []
        for _ in range(0, number_of_neurons):
            neuron_list.append(Neuron(number_of_inputs))
        self.node_layer.append(neuron_list)

    def feed_forward(self, data):
        """Makes prediction based on input data passed"""
        for layer in range(len(self.node_layer)):
            prediction = []
            for row in range(len(data)):
                excited_neurons = []
                for node in range(len(self.node_layer[layer])):
                    neuron = self.node_layer[layer][node]
                    excited_neurons.append(neuron.excite_neuron(data[row]))
                prediction.append(excited_neurons)
            data = prediction
        return prediction

    def classify_predictions(self, predictions):
        new_predictions = []
        for row in range(len(predictions)):
            predict = 0
            high = 0
            for col in range(len(predictions[row])):
                if predictions[row][col] > high:
                    high = predictions[row][col]
                    predict = col
            new_predictions.append(predict)
        return new_predictions


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
    predictions = brain.feed_forward(trainingData)
    print('One layer neural net output')
    print(predictions)
    brain.add_node_layer(3)  # no need to pass second arg if this is not the first node layer
    predictions = brain.feed_forward(trainingData)
    print('Here is the output after adding another layer and running the same data')
    print(predictions)
    predictions = brain.classify_predictions(predictions)
    percent = menu.test(trainingTarget, predictions)
    print("The prediction accuracy before training was %i%%" % percent)

    brain.back_prop(trainingData, trainingTarget)
    predictions = brain.feed_forward(testData)
    predictions = brain.classify_predictions(predictions)
    percent = menu.test(testTarget, predictions)
    print("The prediction accuracy after training was %i%%" % percent)

    """print('Running diabetes data...')
    trainingData, trainingTarget, testData, testTarget = pre_process_diabetes()
    diabetes_net = NeuralNet()
    diabetes_net.add_node_layer(6, len(testData[0]))
    diabetes_net.add_node_layer(2)
    predictions = diabetes_net.feed_forward(testData)
    print('This network has 3 hidden layers')
    predictions = diabetes_net.classify_predictions(predictions)
    percent = menu.test(testTarget, predictions)
    print("The prediction accuracy algorithm was %i%%" % percent)

    diabetes_net.back_prop(testData, testTarget)
    predictions = diabetes_net.feed_forward(testData)
    predictions = diabetes_net.classify_predictions(predictions)
    percent = menu.test(testTarget, predictions)
    print("The prediction accuracy after training was %i%%" % percent)
    """

if __name__ == '__main__':
    main()
