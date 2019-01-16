class Network:
    """Generic Nerual Network Class"""
    # layer is an array containing all layers - input, hidden and output
    # layer[0] is always the input layers, and is just an array of #TODO ints? floats?
    # all other layers are based on a Neuron class
    def __init__(self, numInputs, numOutputs):
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

class Neuron:
    """Generic Neuron Class used by Neural Networks"""
    # weights is an array containing all the weights from the previous layer
    def __init__(self, numWeights):
        self.weights = [0] * numWeights



class Mnist:
    def __init__(self, images, labels):
        self.images = []
        self.labels = []
        with open(images, "rb") as img_file, open(labels, "rb") as lbl_file:
            for count in range(4):
                b = img_file.read(4)
                print(int(b.hex(), 16))
            for count in range(2):
                b = lbl_file.read(4)
                print(int(b.hex(), 16))
            while(True):
                try:
                    self.get_image(img_file)
                    self.get_label(lbl_file)
                    #i = len(self.images)
                    #self.gen_image(self.images[i - 1], self.labels[i - 1]).show()

                except IOError:
                    break
            print("{} images and {} labels".format(len(self.images), len(self.labels)))
    
    def get_label(self, file):
        b = file.read(1)
        if (len(b) != (1)):
            print("Didn't read the whole label!")
            raise IOError
        self.labels.append( int(b.hex()) )

    def get_image(self, file):
        bArray = file.read(28 * 28)
        if (len(bArray) != (28 * 28)):
            print("Didn't read the whole image!")
            raise IOError
        img = [bArray[count] for count in range(len(bArray))]
        self.images.append(img)
        
    def gen_image(self, arr, title):
        from matplotlib import pyplot as plt
        import numpy as np
        two_d = (np.reshape(arr, (28, 28)))
        plt.imshow(two_d, url="huh?", cmap='gray', interpolation='nearest')
        plt.title(str(title))
        return plt            



mnist = Mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte")
#print(mnist.numImages)
