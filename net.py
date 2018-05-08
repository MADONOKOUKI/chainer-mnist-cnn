# -*- mode: python -*-

import chainer
import chainer.functions as F
import chainer.links as L

g_model_filename = 'minst-cnn.model'

# Network definition
class CNN(chainer.Chain):

    def __init__(self):
        super(CNN, self).__init__()
        with self.init_scope():
            self.conv1=L.Convolution2D(1, 20, 5)
            self.conv2=L.Convolution2D(20, 50, 5)
            self.l1=L.Linear(None, 500)
            self.l2=L.Linear(None, 10)

    def __call__(self, x):
        return self.predict(x)

    def predict(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h2 = F.max_pooling_2d(F.relu(self.conv2(h1)), 2)
        h3 = F.relu(self.l1(h2))
        h4 = self.l2(h3)
        return h4
    
# end
