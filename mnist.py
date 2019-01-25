import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(filename)s:%(lineno)s] [%(funcName)s] [%(levelname)s] [%(message)s]')
handler.setFormatter(formatter)
logger.addHandler(handler)
class Mnist:
    def __init__(self, images, labels, testImages, testLabels):
        logger.info("Getting Mnist data setup")
        self.trainImages = []
        self.trainLabels = []
        self.testImages = []
        self.testLabels = []
        with open(images, "rb") as img_file, open(labels, "rb") as lbl_file:
            for count in range(4):
                b = img_file.read(4)
                logger.debug("Train Image Header Info: {}".format(int(b.hex(), 16)))
            for count in range(2):
                b = lbl_file.read(4)
                logger.debug("Train Label Header Info: {}".format(int(b.hex(), 16)))
            while(True):
                try:
                    self.get_image(img_file, self.trainImages)
                    self.get_label(lbl_file, self.trainLabels)
                    #i = len(self.trainImages)
                    #self.gen_image(self.trainImages[i - 1], self.trainLabels[i - 1]).show()

                except IOError:
                    break
            logger.info("TRAINING: {} images and {} labels".format(len(self.trainImages), len(self.trainLabels)))
        with open(testImages, "rb") as img_file, open(testLabels, "rb") as lbl_file:
            for count in range(4):
                b = img_file.read(4)
                logger.debug("Test Image Header Info: {}".format(int(b.hex(), 16)))
            for count in range(2):
                b = lbl_file.read(4)
                logger.debug("Test Image Header Info: {}".format(int(b.hex(), 16)))
            while(True):
                try:
                    self.get_image(img_file, self.testImages)
                    self.get_label(lbl_file, self.testLabels)
                    #i = len(self.trainImages)
                    #self.gen_image(self.trainImages[i - 1], self.trainLabels[i - 1]).show()

                except IOError:
                    break
            logger.info("TEST: {} images and {} labels".format(len(self.testImages), len(self.testLabels)))
    
    def get_label(self, file, labels):
        b = file.read(1)
        if (len(b) != (1)):
            logger.debug("Didn't read the whole label!")
            raise IOError
        labels.append( int(b.hex()) )

    def get_image(self, file, imgs):
        bArray = file.read(28 * 28)
        if (len(bArray) != (28 * 28)):
            logger.debug("Didn't read the whole image!")
            raise IOError
        img = [(bArray[count] / ((255 * 0.99) + 1)) for count in range(len(bArray))]
        imgs.append(img)
        
    def gen_image(self, arr, title):
        from matplotlib import pyplot as plt
        import numpy as np
        two_d = (np.reshape(arr, (28, 28)))
        plt.imshow(two_d, url="huh?", cmap='gray', interpolation='nearest')
        plt.title(str(title))
        return plt            