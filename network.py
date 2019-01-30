import sys
import logging
import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
#handler = logging.FileHandler("log.txt")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)s] [%(funcName)s] [%(levelname)s] [%(message)s]')
handler.setFormatter(formatter)
logger.addHandler(handler)

from scipy.stats import truncnorm
def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd,
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)


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
        self.shape = shape
        logger.debug("Creating network")
        self.learningRate = learningRate
        logger.debug("LearningRate: {}".format(self.learningRate))
        self.layers, self.weights, self.biases = self.create_matricies(shape)
        self.del_weights = [np.zeros(np.array(w).shape) for w in self.weights]
        logger.debug("Total layers: {}".format(len(self.layers)))
        logger.debug("Total bias sets: {}".format(len(self.biases)))
        
    def create_weight_matrix(self, nodes_in, nodes_out):
        #TODO copied from https://www.python-course.eu/neural_network_mnist.php
        rad = 1 / np.sqrt(nodes_in)
        x = truncated_normal(mean=1, sd=1, low=-rad, upp=rad)
        return x.rvs(nodes_in * nodes_out).reshape((nodes_out, nodes_in))
        
    def create_matricies(self, shape):
        layers = []
        biases = []
        weights = []

        for index, layer in enumerate(shape):
            layers.append([0] * layer)
            logger.debug("Layer has {} nodes".format(layer))
            #if (index != 0):
                # We only want weights for the layers other than the input layers
                # TODO Is the bias index appropriate?? biases[0] applies to layer 1?
                #biases.append([0] * layer)
                #logger.debug("Added {} biases for layer {}".format(layer, index))
            # Create weights for every layer but the input layer
            # Each set of weights needs to be a matrix of size [in x out]
            # Thay way we can use dot product of a layer and weights to get
            # to the next layer.
            if(index != len(shape) - 1):
                weights.append(self.create_weight_matrix(shape[index], shape[index + 1]))
                logger.debug("Weight matrix added, size {} by {}".format(shape[index + 1], shape[index]))
        return layers, weights, biases

            
    def start_training(self):
        # During training, we accumulate the changes that each training sample
        #   wants to make to the weights and balances.  When we end the training
        #   session, we apply the #TODO (average of these?) changes all at once
        self.training_samples = 0
        self.del_weights = [w * 0 for w in self.del_weights]
        
#    def accumulate_changes(self, del_weights):
#        self.del_weights += del_weights
#        # TODO biases not used yet.... self.del_biases += del_biases
#        self.training_samples += 1
        
    def end_training(self):
        # When a training set is complete, we want to apply the changes of
        #   each training sample to the weights and biases of the network
        # TODO assume 100 size mini batch
        self.weights += np.array(self.del_weights) / 100

    def train(self, input, target):

        inputVector = np.array(input, ndmin=2).T
        targetVector = np.array(target, ndmin=2).T

        # Feed forward
        hidden_vector = np.dot(self.weights[0], inputVector)
        hidden_output = squishify(hidden_vector)
        output_vector = np.dot(self.weights[1], hidden_output)
        output_network = squishify(output_vector)

        # Backpropagate Output Layer
        output_errors = targetVector - output_network
        tmp = output_errors * output_network * (1.0 - output_network)
        tmp = np.dot(tmp, hidden_output.T)
        self.weights[1] += self.learningRate * tmp

        # Backpropagate Hidden Layer
        output_errors = np.dot(self.weights[1].T, output_errors)
        tmp = output_errors * hidden_output * (1.0 - hidden_output)
        tmp = np.dot(tmp, inputVector.T)
        self.weights[0] += self.learningRate * tmp

    def train_batch(self, input, target):
        # Folloing example on https://www.python-course.eu/neural_network_mnist.php

        inputVector = np.array(input, ndmin=2)
        targetVector = np.array(target, ndmin=2).T
        
        # First, Feed Forward             
        outputVector1 = np.dot(self.weights[0], inputVector.T)
        outputHidden = squishify(outputVector1)
        outputVector2 = np.dot(self.weights[1], outputHidden)
        outputNetwork = squishify(outputVector2)

        # Backpropagate Output Layer
        outputErrors = (targetVector - outputNetwork)
        tmp = outputErrors * outputNetwork * (1.0 - outputNetwork)
        tmp = np.dot(tmp, outputHidden.T)
        self.del_weights[1] += self.learningRate * tmp

        # Backpropagate Hidden Layer
        hiddenErrors = np.dot(self.weights[1].T, outputErrors)
        tmp = hiddenErrors * outputHidden * (1.0 - outputHidden) # This was missing!
        tmp = np.dot(tmp, inputVector)
        self.del_weights[0] += self.learningRate * tmp
            
    def evaluate(self, images, labels):
        corrects, wrongs = 0, 0
        for i in range(len(images)):
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
        input_vector = np.array(input, ndmin=2).T
        output_vector = np.dot(self.weights[0], input_vector)
        output_vector = squishify(output_vector)
        output_vector = np.dot(self.weights[1], output_vector)
        output_vector = squishify(output_vector)
        return output_vector
