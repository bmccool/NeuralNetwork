import sys
import logging
import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)s] [%(funcName)s] [%(levelname)s] [%(message)s]')
handler.setFormatter(formatter)
logger.addHandler(handler)

def sigmoid(x):
    "Squishification"
    return 1 / (1 + np.e ** -x)

class Network:
    def __init__(self, shape, learningRate=0.1):
        logger.info("Creating network")
        self.learningRate = learningRate
        logger.debug("LearningRate: {}".format(self.learningRate))
        self.layers = []
        for layer in shape:
            self.layers.append([0] * layer)
            logger.debug("Layer has {} nodes".format(layer))
        logger.debug("Total layers: {}".format(len(self.layers)))

        self.weights = []
        # Create weights for every layer but the input layer
        # Each set of weights needs to be a matrix of size [in x out]
        # Thay way we can use dot product of a layer and weights to get
        # to the next layer
        for i, layer in enumerate(shape):
            if(i == len(shape) - 1):
                # We are done
                break
            self.weights.append([[0] * shape[i + 1] for each in range(shape[i])])
            logger.debug("Weight matrix added, size {} by {}".format(shape[i + 1], shape[i]))
    
    def train(self, input, target):
        inputVector = np.array(input, ndmin=2)
        targetVector = np.array(target, ndmin=2)

        outputVector1 = np.dot(inputVector, self.weights[0])
        outputHidden = sigmoid(outputVector1)
        outputVector2 = np.dot(outputHidden, self.weights[1])
        outputNetwork = sigmoid(outputVector2)

        # Output error is easy, we know the output and the desired output
        outputErrors = targetVector - outputNetwork

        # Update the weights
        #TODO how does this work? ... Here through the rest of self.train()
        tmp = outputErrors * outputNetwork * (1.0 - outputNetwork)
        tmp = self.learningRate * np.dot(tmp.T, outputHidden)
        self.weights[1] += tmp.T

        # Calculate the hidden layer error
        hiddenErrors = np.dot(self.weights[1], outputErrors.T)

        # Update the weights
        tmp = hiddenErrors.T * outputHidden * (1.0 - outputHidden)
        tmp = self.learningRate * np.dot(tmp.T, inputVector)
        self.weights[0] += tmp.T


    def evaluate(self, images, labels):
        logger.info("Evaluating...")
        corrects, wrongs = 0, 0
        for i in range(len(images)):
            if (i % 1000) == 1:
                logger.info("Evaluating {} / {}".format(corrects, corrects + wrongs))
            res = self.run(images[i])
            resMax = res.argmax()
            if resMax == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs

    def run(self, input):
        """ Run an image through the network """
        # Transpose the input so we have a 1 by X matrix
        # Note, ndmin=2 requires output to be 2d so that dimensions are 1 by X, and not just X
        input_vector = np.array(input, ndmin=2)
        output_vector = np.dot(input_vector, self.weights[0])
        output_vector = sigmoid(output_vector)
        output_vector = np.dot(output_vector, self.weights[1])
        output_vector = sigmoid(output_vector)
        return output_vector



class OldNetwork:
    """Generic Nerual Network Class"""
    # layer is an array containing all layers - input, hidden and output
    # layer[0] is always the input layer, and is just an array of #TODO ints? floats?
    # all other layers are based on a Neuron class
    def __init__(self, numInputs, numOutputs, learningRate=0.1):
        self.learningRate = learningRate
        self.layer = []
        self.layer.append([0] * numInputs)
        #number of neurons in the prevous layer is
        numWeights = len(self.layer[len(self.layer) - 1])
        self.layer.append([Neuron(numWeights) for count in range(numOutputs)])

    def run(self, input):
        """ Run an image through the network """
        # Transpose the input so we have a 1 by X matrix
        # Note, ndmin=2 requires output to be 2d so that dimensions are 1 by X, and not just X
        input_vector = np.array(input, ndmin=2).T

        output_vector = np.dot()


    def train(self, input, target):
        logger.info("Training...")

    def evaluate(self, images, labels):
        logger.info("Evaluating...")
        return 0, len(images)

    def addLayer(self, numNeurons):
        """
        Add a new layer full of neurons.  All layers are added just prior to
        the output stage.
        """
        #number of neurons in the next to last layer is
        numWeights = len(self.layer[len(self.layer) - 2])
        self.layer.insert(len(self.layer) - 1, [Neuron(numWeights) for count in range(numNeurons)])

        #Now we need to update the weights for the last layer
        #number of neurons in the next to last layer is
        numWeights = len(self.layer[len(self.layer) - 2])
        self.layer[len(self.layer) - 1] = [Neuron(numWeights) for count in range(len(self.layer[len(self.layer) - 1]))]

    def status(self):
        for index, layer in enumerate(self.layer):
            logger.info("Layer {} has {} nodes".format(index, len(layer)))
            if index > 0:
                for node in layer:
                    if len(node.weights) != len(self.layer[index - 1]):
                        logger.info("Error! Node in layer {} has {} weights instead \
                               of {}".format(index, len(node.weights), len(self.layer[index - 1])))

class Neuron:
    """Generic Neuron Class used by Neural Networks"""
    # weights is an array containing all the weights from the previous layer
    def __init__(self, numWeights):
        self.weights = [0] * numWeights
