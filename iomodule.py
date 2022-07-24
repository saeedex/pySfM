from config import *

class images:
    def __init__(self):
        self.color = []
        self.gray = []

    def load(self, config):
        os.chdir(config.dataset.path)
        for i in glob.glob("*.jpg"):
            img = cv2.imread(config.dataset.path + i)
            img = cv2.resize(img, (int(img.shape[1] / config.dataset.imresize),
                                   int(img.shape[0] / config.dataset.imresize)))
            gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.color.append(img)
            self.gray.append(gimg)

            if config.dataset.display:
                cv2.imshow('sample image', img)
                cv2.waitKey(1)