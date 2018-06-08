'''
Created on 24 mars. 2018

@author: John Henry Dahlberg
'''

from numpy import *
import matplotlib.pyplot as plt
from sklearn.neural_network import BernoulliRBM
import importlib
import sklearn.preprocessing

mlpImported = importlib.__import__('MLP')
mlpTensorFlowImported = importlib.__import__('neuralNetwork')


import copy


def restrictedBoltzmannMachine(xTrain, xTest, nHiddenNeurons, nEpochs, learningRate, plotErrors=False, plotWeights=False):
    Ntrain = xTrain.shape[0]
    Ntest = xTest.shape[0]

    if plotErrors:
        msErrorsTrain = zeros(nEpochs)
        msErrorsTest = zeros(nEpochs)
        for iteration in range(nEpochs):
            rbm = BernoulliRBM(n_components=nHiddenNeurons, learning_rate=learningRate, random_state=1, verbose=True, n_iter=iteration + 1)
            rbm.fit(xTrain)
            XRecoveredTrain = rbm.gibbs(xTrain)
            XRecoveredTest = rbm.gibbs(xTest)
            msErrorsTrain[iteration] = sum(square(xTrain - XRecoveredTrain)) / Ntrain
            msErrorsTest[iteration] = sum(square(xTest - XRecoveredTest)) / Ntest
            print('Mean square error train at epoch', iteration + 1, ': ', msErrorsTrain[iteration])
            print('Mean square error test at epoch', iteration + 1, ': ', msErrorsTest[iteration])

        plt.plot(msErrorsTrain, label='Train')
        plt.plot(msErrorsTest, label='Test')
        plt.legend(loc='best')
        plt.xlabel('Epochs')
        plt.ylabel('MSE - original and recovered pattern')
        plt.title('RBM \nnHiddenNeurons = ' + str(nHiddenNeurons));

    else:
        rbm = BernoulliRBM(n_components=nHiddenNeurons, learning_rate=learningRate, random_state=1, verbose=True,n_iter=nEpochs)
        rbm.fit(xTrain)
        XRecoveredTrain = rbm.gibbs(xTrain)
        XRecoveredTest = rbm.gibbs(xTest)
        msErrorsTrain = sum(square(xTrain - XRecoveredTrain)) / Ntrain
        print('Mean square error(Trainset):', msErrorsTrain)
        msErrorsTest = sum(square(xTest - XRecoveredTest)) / Ntest
        print('Mean square error(Testset):', msErrorsTest)

    if plotWeights:
        plt.figure(figsize=(20, 20))
        nGrid = int(sqrt(nHiddenNeurons))
        plt.tight_layout()
        for i, comp in enumerate(rbm.components_):
            plt.subplot(nGrid, nGrid, i + 1)
            plt.imshow(comp.reshape((28, 28)), cmap=plt.cm.RdYlGn,
                       interpolation='nearest', vmin=-2.5, vmax=2.5)
            plt.margins(tight=True)
            plt.axis('off')
        plt.suptitle('Weights of all ' + str(nHiddenNeurons) + ' hidden neurons visualized')


    return XRecoveredTest, msErrorsTest

def recoverClassDigits(xTrain, xTest, targetTrain, targetTest, nHiddenNeurons, nEpochs, learningRate):

    XRecoveredTest, msErrorsTest = restrictedBoltzmannMachine(xTrain, xTest, nHiddenNeurons, nEpochs, learningRate, plotWeights=True)

    nClasses = 10
    indicesDigitClasses = zeros(nClasses)
    digitClass = 0
    for i in range(len(targetTest)):
        if targetTest[i] == digitClass:
            indicesDigitClasses[digitClass] = i
            digitClass += 1
        if digitClass == nClasses:
            break

    xTestImages = xTest.reshape(-1, 28, 28)
    xTestRecoveredImages = XRecoveredTest.reshape(-1, 28, 28)

    plt.figure()
    for i in range(nClasses):
        plt.subplot(2, nClasses, i + 1)
        plt.imshow(xTestImages[int(indicesDigitClasses[i])])
        plt.axis('off')

        plt.subplot(2, nClasses, nClasses + i + 1)
        plt.imshow(xTestRecoveredImages[int(indicesDigitClasses[i])])
        plt.axis('off')

        plt.suptitle("Original and its recovered digit for each class, \nnHiddenNeurons = " + str(nHiddenNeurons) + ",\n nEpochs = " + str(nEpochs) + ",\nMean square error = " + str(msErrorsTest))


def preTrain(xTrain, xTest, inputLayerSize, outputLayerSize, hiddenLayerSizes, nEpochs, learningRate):
    print('Pre-training weights with RBM')

    layerNeuronSizes = copy.copy(hiddenLayerSizes)
    layerNeuronSizes.insert(0, inputLayerSize)
    layerNeuronSizes.append(outputLayerSize)
    nLayers = len(layerNeuronSizes)
    inputPattern = xTrain
    inputPatternTest = xTest
    W = []
    b = []

    for layer in range(1, nLayers):
        nHiddenNeuronsRBM = layerNeuronSizes[layer]
        rbm = BernoulliRBM(n_components=nHiddenNeuronsRBM, learning_rate=learningRate, random_state=1, verbose=True, n_iter=nEpochs)
        rbm.fit(inputPattern)
        W.append(array(rbm.components_).T)
        b.append(array(rbm.intercept_hidden_).T)
        outputPattern = rbm.gibbs(inputPattern)

        N = xTest.shape[0]
        xTestRecovered = rbm.gibbs(inputPatternTest)
        msErrorsTrain = sum(square(inputPatternTest - xTestRecovered)) / N
        print('Mean square test error after training layer ' + str(layer) + ': ', msErrorsTrain)

        inputPattern = rbm.transform(inputPattern)
        inputPatternTest = rbm.transform(inputPatternTest)

    return W, b


def main():
    xTrain = genfromtxt('bindigit_trn.csv', delimiter=',')
    targetTrain = genfromtxt('targetdigit_trn.csv', delimiter=',')
    xTest = genfromtxt('bindigit_tst.csv', delimiter=',')
    targetTest = genfromtxt('targetdigit_tst.csv', delimiter=',')

    nEpochs = 1
    learningRate = 0.25

    # Square numbers to plot patterns in squared grid
    nHiddenNeurons = 225

    recoverClassDigits(xTrain, xTest, targetTrain, targetTest, nHiddenNeurons, nEpochs, learningRate)


if __name__ == '__main__':
    main()
    plt.show()
