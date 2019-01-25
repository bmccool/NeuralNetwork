import sys
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)s] [%(funcName)s] [%(levelname)s] [%(message)s]')
handler.setFormatter(formatter)
logger.addHandler(handler)
class Network:
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

    def train(self, image, label):
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