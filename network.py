import sys
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
        print(numWeights)
        self.layer.append([Neuron(numWeights) for count in range(numOutputs)])

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
        maxRows = max([len(layer) for layer in self.layer])
        for row in range(maxRows):
            for col in range(len(self.layer)):
                if(len(self.layer[col])) > row:
                    sys.stdout.write(str(col).rjust(4, " "))
                    #sys.stdout.write(str(self.layer[col][row]).rjust(5, " "))
            sys.stdout.write("\n")

class Neuron:
    """Generic Neuron Class used by Neural Networks"""
    # weights is an array containing all the weights from the previous layer
    def __init__(self, numWeights):
        self.weights = [0] * numWeights