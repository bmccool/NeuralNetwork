from mnist import Mnist


def one_hot(label, total=10):
    lr = np.arange(total)
    one_hot_label = (lr == label).astype(np.int)
    one_hot_label[one_hot_label == 0] = 0.01
    one_hot_label[one_hot_label == 1] = 0.99





#mnist = Mnist("train-images.idx3-ubyte", "train-labels.idx1-ubyte")

epochs = 3

from network import Network

nn = Network(28 * 28, 10)
nn.addLayer(100)

nn.status()