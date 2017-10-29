from network_layers import *
from transfer_functions import *

class NeuralNetwork(object):
    """
    A class which streamlines the output through layers and error calculation.
    """

    def __init__(self):
        """
        Initializes the layers of the neural network to an empty array.
        """
        self.layers = []

    def addLayer(self, layer):
        """
        Adds a layer to the neural network in sequential fashion.
        """
        self.layers.append(layer)

    def output(self, x):
        """
        Calculates the output for a single input instance x (one row from
        the training or test set)
        """
        values = x
        for layer in self.layers:
            values = layer.output(values)
        return values

    def outputs(self, X):
        """
        For a given vector of input instances X (the training or test set),
        returns the vector of outputs for all the input instances.
        """
        result = []
        for values in X:
            result.append(self.output(values))
        return result

    def error(self, prediction, y):
        """
        Calculates the square error for a single example in the train/test set.
        """
        return (prediction - y) * (prediction - y)

    def total_error(self, predictions, Y):
        """
        Calculates the mean square error for all the examples in the train/test set.
        """
        error = 0
        for prediction, y in zip(predictions, Y):
            error += self.error(prediction, y)
        return 1.0 * error / len(predictions)

    def forwardStep(self, X, Y):
        """
        Runs the inputs X (train/test set) through the network, and calculates
        the error on the given true target function values Y.
        """
        return self.total_error(self.outputs(X), Y)

    def size(self):
        """
        Returns the total number of weights in the network.
        """
        totalSize = 0
        for layer in self.layers:
            totalSize += layer.size()
        return totalSize

    def getWeightsFlat(self):
        """
        Returns a 1-d representation of all the weights in the network. First
        layer comes first in the flat vector.
        """
        flatWeights = np.array([])
        for layer in self.layers:
            flatWeights = np.append(flatWeights, layer.getWeightsFlat())
        return flatWeights

    def setWeights(self, flat_vector):
        """
        Sets the weights for all layers in the network. First layer comes
        first in the flat vector.
        """
        for layer in self.layers:
            layer.setWeights(flat_vector[:layer.size()])
            flat_vector = flat_vector[layer.size():]
