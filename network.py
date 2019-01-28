import sys
import logging
import numpy as np
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
#handler = logging.StreamHandler()
handler = logging.FileHandler("log.txt")
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
        logger.info("Creating network")
        self.learningRate = learningRate
        logger.debug("LearningRate: {}".format(self.learningRate))
        #for index, layer in enumerate(shape):
        #    self.layers.append([0] * layer)
        #    logger.debug("Layer has {} nodes".format(layer))
        #    if (index != 0):
        #        # We only want weights for the layers other than the input layers
        #        # TODO Is the bias index appropriate?? biases[0] applies to layer 1?
        #        self.biases.append([0] * layer)
        #        logger.debug("Added {} weights for layer {}".format(layer, index))
        #    # Create weights for every layer but the input layer
        #    # Each set of weights needs to be a matrix of size [in x out]
        #    # Thay way we can use dot product of a layer and weights to get
        #    # to the next layer.
        #    if(i != len(shape) - 1):
        #        self.weights.append([[0] * shape[i + 1] for each in range(shape[i])])
        #        logger.debug("Weight matrix added, size {} by {}".format(shape[i + 1], shape[i]))
        self.layers, self.weights, self.biases = self.create_matricies(shape)
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
            #logger.debug("Layer has {} nodes".format(layer))
            if (index != 0):
                # We only want weights for the layers other than the input layers
                # TODO Is the bias index appropriate?? biases[0] applies to layer 1?
                biases.append([0] * layer)
                #logger.debug("Added {} biases for layer {}".format(layer, index))
            # Create weights for every layer but the input layer
            # Each set of weights needs to be a matrix of size [in x out]
            # Thay way we can use dot product of a layer and weights to get
            # to the next layer.
            if(index != len(shape) - 1):
                weights.append(self.create_weight_matrix(shape[index], shape[index + 1]))
                #logger.debug("Weight matrix added, size {} by {}".format(shape[index + 1], shape[index]))
        return layers, weights, biases

            
    def start_training(self):
        # During training, we accumulate the changes that each training sample
        #   wants to make to the weights and balances.  When we end the training
        #   session, we apply the #TODO (average of these?) changes all at once
        self.training_samples = 0
        layers, self.del_weights, self.del_biases = self.create_matricies(self.shape)
        self.del_weights = np.array(self.del_weights)
        self.del_biases = np.array(self.del_biases)
        #logger.info("Starting training session")
        
    def accumulate_changes(self, del_weights, del_biases):
        self.del_weights += del_weights
        # TODO biases not used yet.... self.del_biases += del_biases
        self.training_samples += 1
        
    def end_training(self):
        # When a training set is complete, we want to apply the changes of
        #   each training sample to the weights and biases of the network
        self.del_weights = np.array(self.del_weights) / self.training_samples
        # For some reason, biases is list instead of array, can't operate list and int
        #self.del_biases = np.array(self.del_biases) / self.training_samples
        self.weights += self.del_weights
        #self.biases += self.del_biases

    def train_single(self, input_vector, target_vector):
        # input_vector and target_vector can be tuple, list or ndarray
                                       
        no_of_layers = 3       
        input_vector = np.array(input_vector, ndmin=2).T
        layer_index = 0
        # The output/input vectors of the various layers:
        res_vectors = [input_vector]          
        while layer_index < no_of_layers - 1:
            in_vector = res_vectors[-1]

            x = np.dot(self.weights[layer_index], in_vector)
            out_vector = squishify(x)
            res_vectors.append(out_vector)   
            layer_index += 1
        
        layer_index = no_of_layers - 1
        target_vector = np.array(target_vector, ndmin=2).T
         # The input vectors to the various layers
        output_errors = target_vector - out_vector  
        while layer_index > 0:
            out_vector = res_vectors[layer_index]
            in_vector = res_vectors[layer_index-1]
            tmp = output_errors * out_vector * (1.0 - out_vector)     
            tmp = np.dot(tmp, in_vector.T)
                       
            self.weights[layer_index-1] += self.learningRate * tmp
            
            output_errors = np.dot(self.weights[layer_index-1].T, 
                                   output_errors)
            layer_index -= 1
        
            
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
        targetVector = np.array(target, ndmin=2).T

        outputVector1 = np.dot(self.weights[0], inputVector.T)
        outputHidden = squishify(outputVector1)
        outputVector2 = np.dot(self.weights[1], outputHidden)
        outputNetwork = squishify(outputVector2)

        # Backpropagate Output Layer
        
        # We need the error multiplied by the derivative of the activation function
        # the derivative of the sigmoix(x) is x * (1 - x)
        # so we want (target - output) output * (1 - output)
        logger.debug("Target: {}".format(targetVector))
        logger.debug("Output: {}".format(outputNetwork))
        outputErrors = (targetVector - outputNetwork)
        # outputErrors is 10x10... what should it be?
        logger.debug("Output Errors: {}".format(outputErrors))
        # Update the weights
        # We want weights = weights + learning rate * Error * Input
        #logger.debug("np.dot(outputErrors.T, outputHidden).T = {}".format(dot))
        #                                      This dot makes #100x10
        tmp = outputErrors * outputNetwork * (1.0 - outputNetwork)
        #            (10,1) and (100,1)
        tmp = np.dot(tmp, outputHidden)
        del_weights = [np.zeros(np.array(w).shape) for w in self.weights]
        del_weights[1] = self.learningRate * tmp

        # Backpropagate Hidden Layer
        # Error for a hidden layer node is the weighted sum of all the connected nodes,
        # together with the derivative of the activation function
        # I.e. Error(Node[j]) = SUM[1-10](weights[1][j][1-10] * outputError[1-10]) * outputHidden * (1 - outputHidden)
        hiddenErrors = np.dot(self.weights[1], outputErrors.T).T * outputHidden * (1.0 - outputHidden)
        logger.debug("HiddenErrors {}".format(hiddenErrors))
        # Update the weights
        del_weights[0] = self.learningRate * np.dot(hiddenErrors.T, inputVector).T
        
        for i, weights in enumerate(del_weights):
            for j, row in enumerate(weights):
                logger.debug("Weights[{}][{}]: {}".format(i, j, row))
        raise "STOP"
        
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
            logger.debug("Label: {}, RES: {}".format(labels[i], res))
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