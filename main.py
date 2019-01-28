from mnist import Mnist
import numpy as np
import logging

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


logger.debug("START")
mnist = Mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte",
               "t10k-images.idx3-ubyte",  "t10k-labels.idx1-ubyte")

epochs = 1
testPercent = []
trainingPercent = []

from network import Network

nn = Network([28 * 28, 100, 10], 0.2)

for epoch in range(epochs):
    logger.info("epoch {}".format(epoch))
    nn.start_training()
    for i in range(len(mnist.trainImages)):
        #nn.train(mnist.trainImages[i], one_hot(mnist.trainLabels[i]))
        nn.train_single(mnist.trainImages[i], one_hot(mnist.trainLabels[i]))
        if (i % 10000 ) == 0:
            logger.info("Trained {} / {}".format(i, len(mnist.trainImages)))
            #nn.end_training()
            #raise "STOP"
            #nn.start_training()

    corrects, wrongs = nn.evaluate(mnist.trainImages, mnist.trainLabels)
    logger.info("{:.2f}% Correct in training data".format((corrects / (corrects + wrongs)) * 100))
    trainingPercent.append("{:.2f}%".format((corrects / (corrects + wrongs)) * 100))

    corrects, wrongs = nn.evaluate(mnist.testImages, mnist.testLabels)
    logger.info("{:.2f}% Correct in test data".format((corrects / (corrects + wrongs)) * 100))
    testPercent.append("{:.2f}%".format((corrects / (corrects + wrongs)) * 100))
    
logger.info("Training Percent: {}".format(trainingPercent))
logger.info("Test Percent: {}".format(testPercent))
