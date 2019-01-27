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
  
def d_sigmoid(x):
    return (x * (1.0 - x))
  
def squishify(x):
    return sigmoid(x)
  
def d_squishify(x):
    return d_sigmoid(x)


class Network:
    def __init__(self, shape, learningRate=0.1):
        logger.info("Creating network")
        self.learningRate = learningRate
        logger.debug("LearningRate: {}".format(self.learningRate))
        self.layers = []
        self.biases = []
        for index, layer in enumerate(shape):
            self.layers.append([0] * layer)
            logger.debug("Layer has {} nodes".format(layer))
            if (index != 0):
                # We only want weights for the layers other than the input layers
                # TODO Is the bias index appropriate?? biases[0] applies to layer 1?
                self.biases.append([0] * layer)
                logger.debug("Added {} weights for layer {}".format(layer, index))
        logger.debug("Total layers: {}".format(len(self.layers)))
        logger.debug("Total bias sets: {}".format(len(self.biases)))

        self.weights = []
        # Create weights for every layer but the input layer
        # Each set of weights needs to be a matrix of size [in x out]
        # Thay way we can use dot product of a layer and weights to get
        # to the next layer.
        for i, layer in enumerate(shape):
            if(i == len(shape) - 1):
                # We are done
                break
            self.weights.append([[0] * shape[i + 1] for each in range(shape[i])])
            logger.debug("Weight matrix added, size {} by {}".format(shape[i + 1], shape[i]))
            
    def start_training(self):
        # During training, we accumulate the changes that each training sample
        #   wants to make to the weights and balances.  When we end the training
        #   session, we apply the #TODO (average of these?) changes all at once
        self.del_weights = self.weights.copy()
        self.del_biases = self.biases.copy()
        self.del_weights = np.array(self.del_weights) * 0
        logger.warn("WEIGHTS:")
        logger.warn(self.del_weights[0])
        # TODO left here
        # Why on earth can I not zero the del_weights matrix?  It just keeps disappearing.
        self.del_biases = np.dot(np.array(self.del_biases), 0)
        self.training_samples = 0
        logger.info("Starting training session")
        
    def accumulate_changes(self, del_weights, del_biases):
        logger.warn(len(del_weights))
        self.del_weights += del_weights
        # TODO biases not used yet.... self.del_biases += del_biases
        self.training_samples += 1
        
    def end_training(self):
        # When a training set is complete, we want to apply the changes of
        #   each training sample to the weights and biases of the network
        self.weights += self.del_weights
        self.biases += self.del_biases
            
    def train(self, input, target):
        # Folloing example on https://www.python-course.eu/neural_network_mnist.php
        # First, Feed Forward
        # The output (activation) of the network is the output of the last layer.
        # The activation of each neuron in a layer is the weighted sum of the
        #     activations of all neurons in the previous layer, plus a bias,
        #     and the whole thing run through a "squishification" function to
        #     distribute the result between [0-1].
        #
        inputVector = np.array(input, ndmin=2)
        targetVector = np.array(target, ndmin=2)

        outputVector1 = np.dot(inputVector, self.weights[0])
        outputHidden = squishify(outputVector1)
        outputVector2 = np.dot(outputHidden, self.weights[1])
        outputNetwork = squishify(outputVector2)

        # Backpropagate Output Layer
        
        # We need the error multiplied by the derivative of the activation function
        # the derivative of the sigmoix(x) is x * (1 - x)
        # so we want (target - output) output * (1 - output)
        outputErrors = (targetVector - outputNetwork) * outputNetwork * (1.0 - outputNetwork)
        # Update the weights
        # We want weights = weights + learning rate * Error * Input
        #logger.debug("np.dot(outputErrors.T, outputHidden).T = {}".format(dot))
        #                                      This dot makes #100x10
        del_weights = [[], []]
        del_weights[1] = self.learningRate * np.dot(outputErrors.T, outputHidden).T

        # Backpropagate Hidden Layer
        # Error for a hidden layer node is the weighted sum of all the connected nodes,
        # together with the derivative of the activation function
        # I.e. Error(Node[j]) = SUM[1-10](weights[1][j][1-10] * outputError[1-10]) * outputHidden * (1 - outputHidden)
        hiddenErrors = np.dot(self.weights[1], outputErrors.T).T * outputHidden * (1.0 - outputHidden)
        # Update the weights
        del_weights[0] = self.learningRate * np.dot(hiddenErrors.T, inputVector).T
        self.accumulate_changes(del_weights, 0)
        
        
        
#    def trainIndividual(self, input, target):
#        #TODO Assumes a lot about network size
#        # https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/
#        inputVector = np.array(input, ndmin=2)
#        targetVector = np.array(target, ndmin=2)
#
#        outputVector1 = np.dot(inputVector, self.weights[0])
#        outputHidden = sigmoid(outputVector1)
#        outputVector2 = np.dot(outputHidden, self.weights[1])
#        outputNetwork = sigmoid(outputVector2)
#        
#        # Backpropagate Output Layer
#        # We need the error multiplied by the derivative of the activation function
#        # the derivative of the sigmoix(x) is x * (1 - x)
#        # so we want (target - output) output * (1 - output)       
#        for i in reversed(range(len(self.layers))):
#            layer = self.layers[i]
#            errors = list()
#            if i != len(self.layers) - 1:
#                for j in range(len(layer)):
#                    error = 0.0
#                    for neuron, value in enumerate(self.layers[i + 1]):
#                        error += (weights[1][j][neuron] * (target[neuron] - outputNetwork[neuron]) * d_sigmoid(outputNetwork[neuron]))
#                    errors.append(error)
#            else:
#                for j in range(len(layer)):
#                    neuron = layer[j]
#                    errors.append(target[])
#                    #FINISH


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
        output_vector = squishify(output_vector)
        output_vector = np.dot(output_vector, self.weights[1])
        output_vector = squishify(output_vector)
        return output_vector