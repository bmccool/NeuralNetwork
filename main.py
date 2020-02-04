from mnist import Mnist
import numpy as np
import logging
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
#handler = logging.FileHandler("log.txt")
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)s] [%(funcName)s] [%(levelname)s] [%(message)s]')
handler.setFormatter(formatter)
logger.addHandler(handler)


import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        ## 1 input image channel, 6 output channels, 3x3 square convolution
        ## kernel
        #self.conv1 = nn.Conv2d(1, 6, 3)
        #self.conv2 = nn.Conv2d(6, 16, 3)
        ## an affine operation: y = Wx + b
        #self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        #self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)
        # We want the network to have a shape of [28 * 28, 100, 10]
        self.stage1 = nn.Linear(28 * 28, 100)
        self.stage2 = nn.Linear(100, 10)

    def forward(self, x):
        ## Max pooling over a (2, 2) window
        #x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        ## If the size is a square you can only specify a single number
        #x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        #x = x.view(-1, self.num_flat_features(x))
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        #x = self.fc3(x)
        x = F.relu(self.stage1(x))
        # BAM: why no relu in output stage?
        x = self.stage2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def one_hot(label, total=10):
    import numpy as np
    lr = np.arange(total)
    one_hot_label = (lr == label).astype(np.float)
    one_hot_label[one_hot_label == 0] = 0.01
    one_hot_label[one_hot_label == 1] = 0.99
    return one_hot_label
  
def evaluate(mnist_data, network):
    #corrects, wrongs = network.evaluate(mnist.trainImages, mnist.trainLabels)
    #logger.info("{:.2f}% Correct in training data".format((corrects / (corrects + wrongs)) * 100))
    #trainingPercent.append("{:.2f}%".format((corrects / (corrects + wrongs)) * 100))

    corrects, wrongs = network.evaluate(mnist.testImages, mnist.testLabels)
    #logger.info("{:.2f}% Correct in test data".format((corrects / (corrects + wrongs)) * 100))

    return "{:.2f}%".format((corrects / (corrects + wrongs)) * 100)

def torch_eval(net):
    #logger.debug("Evaluating network")
    corrects, wrongs = 0, 0
    for i in range(len(mnist.testImages)):
        res = net(torch.tensor(mnist.testImages[i]).unsqueeze(0).unsqueeze(0))
        resMax = res.argmax()
        if resMax == mnist.testLabels[i]:
            corrects += 1
        else:
            wrongs += 1
        #if (i % 1000 ) == 0:
        #    logger.debug("Tested {} / {}...".format(i, len(mnist.testImages)))
    return "{:.2f}%".format((corrects / (corrects + wrongs)) * 100)

logger.debug("START")
mnist = Mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
               "t10k-images.idx3-ubyte",  "t10k-labels.idx1-ubyte")
shuffledList = list(range(len(mnist.trainImages)))
random.shuffle(shuffledList)

epochs = 2 
 
testPercent = []
trainingPercent = []

from network import Network
import torch.optim as optim

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
    #nn_batch.start_training()
    for i in shuffledList:
        # gather all the samples in this batch
        batch_images_list.append(torch.tensor(mnist.trainImages[i]).unsqueeze(0))
        label = torch.from_numpy(one_hot(mnist.trainLabels[i]))
        label = label.view(1, -1) # why is this here?
        #label = label.unsqueeze(0)
        label = label.type(torch.FloatTensor)
        batch_labels_list.append(label)

        if ((shuffledList.index(i) % 100)  == 0) and (shuffledList.index(i) != 0):
            # Batch is gathered!
            # Train the network for this batch

            # Zero the gradient buffers
            optimizer.zero_grad()

            # Feed forward
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
