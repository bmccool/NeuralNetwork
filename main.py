from mnist import Mnist
import numpy as np
import logging
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#TODO fix this mess...
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
#handler = logging.FileHandler("log.txt")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)s] [%(funcName)s] [%(levelname)s] [%(message)s]')
handler.setFormatter(formatter)
logger.addHandler(handler)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # We want the network to have a shape of [28 * 28, 100, 10]
        # This means a 28x28 input image, hidden layer of 100, output layer of 10
        self.stage1 = nn.Linear(28 * 28, 100)
        self.stage2 = nn.Linear(100, 10)

    def forward(self, x):
        # Feed x forward through the network
        # TODO relu is the activation function, why do we not use it on the output layer?
        x = F.relu(self.stage1(x))
        x = self.stage2(x)
        return x


def one_hot(label, total=10):
    # Take an integer as label and return one hot representation
    # label=6 -> [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    import numpy as np
    lr = np.arange(total)
    one_hot_label = (lr == label).astype(np.float)
    one_hot_label[one_hot_label == 0] = 0.01
    one_hot_label[one_hot_label == 1] = 0.99
    return one_hot_label
  
def torch_eval(net):
    # Check the network against the training set.
    # Return %correct to be logged
    corrects, wrongs = 0, 0
    for i in range(len(mnist.testImages)):
        res = net(torch.tensor(mnist.testImages[i]).unsqueeze(0).unsqueeze(0))
        resMax = res.argmax()
        if resMax == mnist.testLabels[i]:
            corrects += 1
        else:
            wrongs += 1
    return "{:.2f}%".format((corrects / (corrects + wrongs)) * 100)

logger.debug("START")

# Import mnist training and test images and labels
mnist = Mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
               "t10k-images.idx3-ubyte",  "t10k-labels.idx1-ubyte")
               
# Shuffle training set
shuffledList = list(range(len(mnist.trainImages)))
random.shuffle(shuffledList)

# epoch is one entire run through the training set
epochs = 2 

# create the network
nn_torch = Net()

# Create optimizer
optimizer = optim.SGD(nn_torch.parameters(), lr=0.02)
criterion = nn.MSELoss()

logger.info("Using this torch net:")
logger.info(nn_torch)

batch_size = 100
batch_images_list = []
batch_labels_list = []

for epoch in range(epochs):
    logger.info("epoch {}".format(epoch))
    for i in shuffledList:
        # Gather the training images
        # Unsqueeze just pads an extra dimension to a matrix. [2][2] -> [1][2][2]
        # This is needed because torch wants a tensor of [batch size][channels][input]
        #  and we don't have separate channels for this example
        batch_images_list.append(torch.tensor(mnist.trainImages[i]).unsqueeze(0))
        
        # Gather the labels that go with those images in "one hot" format
        label = torch.from_numpy(one_hot(mnist.trainLabels[i]))
        label = label.view(1, -1) # rotate label
        label = label.type(torch.FloatTensor)
        batch_labels_list.append(label)

        if ((shuffledList.index(i) % batch_size)  == 0) and (shuffledList.index(i) != 0):
            # This batch is full!  Start training...

            # Zero the gradient buffers
            optimizer.zero_grad()

            # Feed training images forward
            batch_images = torch.stack([*batch_images_list])
            output = nn_torch(batch_images)
            batch_images_list = []

            # Calculate Loss
            batch_labels = torch.stack([*batch_labels_list])
            loss = criterion(output, batch_labels)
            batch_labels_list = []
            
            # Backpropagate
            loss.backward()
     
            # Update parameters
            optimizer.step()

            # Evaluate after every batch
            logger.info("Epoch: {}, {} / {}, torchNet: {}".format(epoch+1, shuffledList.index(i), len(mnist.trainImages), torch_eval(nn_torch)))

    logger.info("After Epoch {}, torchNet: {}".format(epoch+1, torch_eval(nn_torch)))