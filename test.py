#!/usr/bin/env python

import argparse

import chainer
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from net import CNN, g_model_filename

def print_predict(model, dir):
    # Load the MNIST dataset
    _, test = chainer.datasets.get_mnist(ndim=3)
    test, label = chainer.dataset.concat_examples(test)

    pred = model.predict(test)
    pred = chainer.functions.softmax(pred).data
    label_y = [np.argmax(pred[i]) for i in range(len(pred))]
    
    result = confusion_matrix(label, label_y)
        
    print('     0    1    2    3    4    5    6    7    8    9   accuracy        ')
        
    for y in range(10):
        print('{}'.format(y), end='')
        for x in range(10):
            print('{0:5d}'.format(result[y][x]), end='')
        print('   {0:1.5f}         '.format(result[y][y] / np.sum(result[y])))

    print('accuracy all: {0:1.5f}'.format(accuracy_score(label, label_y)))


def main():
    parser = argparse.ArgumentParser(description='regression of kWh')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    args = parser.parse_args()

    model = CNN()

    # load model
    chainer.serializers.load_npz(args.out + '/' + g_model_filename, model)

    print_predict(model, args.out)


if __name__ == '__main__':
    main()

# debug
# import pdb; pdb.set_trace()

## end ##
