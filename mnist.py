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
        img = [(bArray[count] / ((255 * 0.99) + 1)) for count in range(len(bArray))]
        self.images.append(img)
        
    def gen_image(self, arr, title):
        from matplotlib import pyplot as plt
        import numpy as np
        two_d = (np.reshape(arr, (28, 28)))
        plt.imshow(two_d, url="huh?", cmap='gray', interpolation='nearest')
        plt.title(str(title))
        return plt            