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


logger.debug("START")
mnist = Mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
               "t10k-images.idx3-ubyte",  "t10k-labels.idx1-ubyte")
shuffledList = list(range(len(mnist.trainImages)))
random.shuffle(shuffledList)

epochs = 1
 
testPercent = []
trainingPercent = []

from network import Network

nn_batch = Network([28 * 28, 100, 10], 0.2)
nn_single = Network([28 * 28, 100, 10], 0.2)

for epoch in range(epochs):
    logger.info("epoch {}".format(epoch))
    nn_batch.start_training()
    for i in shuffledList:
        nn_batch.train_batch(mnist.trainImages[i], one_hot(mnist.trainLabels[i]))
        nn_single.train(mnist.trainImages[i], one_hot(mnist.trainLabels[i]))
        if (shuffledList.index(i) % 100 ) == 0:
            nn_batch.end_training()
            logger.info("Epoch: {}, {} / {}, Batch: {}, Single: {}".format(epoch+1,
                        shuffledList.index(i), len(mnist.trainImages), evaluate(mnist, nn_batch), evaluate(mnist, nn_single)))
            nn_batch.start_training()

    nn_batch.end_training()
