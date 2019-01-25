from mnist import Mnist
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)s] [%(funcName)s] [%(levelname)s] [%(message)s]')
handler.setFormatter(formatter)
logger.addHandler(handler)


def one_hot(label, total=10):
    import numpy as np
    lr = np.arange(total)
    one_hot_label = (lr == label).astype(np.int)
    one_hot_label[one_hot_label == 0] = 0.01
    one_hot_label[one_hot_label == 1] = 0.99


logger.debug("START")
mnist = Mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
               "t10k-images.idx3-ubyte",  "t10k-labels.idx1-ubyte")

epochs = 1

from network import Network

nn = Network(28 * 28, 10)
nn.addLayer(100)

nn.status()

for epoch in range(epochs):
    logger.info("epoch {}".format(epoch))
    for i in range(len(mnist.trainImages)):
	    nn.train(mnist.trainImages[i], one_hot(mnist.trainLabels[i]))
    corrects, wrongs = nn.evaluate(mnist.trainImages, mnist.trainLabels)
    logger.info("{:.2f}% Correct in training data".format(corrects / (corrects + wrongs)))
    corrects, wrongs = nn.evaluate(mnist.testImages, mnist.testLabels)
    logger.info("{:.2f}% Correct in test data".format(corrects / (corrects + wrongs)))