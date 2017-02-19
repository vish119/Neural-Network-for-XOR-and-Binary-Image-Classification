# sample_submission.py
import numpy as np
from scipy.special import expit
import sys


class xor_net(object):
    """
    This code will train and test  the Neural Network for XOR data.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.

    """
    def __init__(self, data, labels):
        self.x = data
        self.y = labels
        maxiteration = 300000
        if self.x.shape[0] <= 100:
            learningrate = .001
            maxiteration = 1000000
        elif self.x.shape[0] <= 500:
            learningrate = .0001
            maxiteration = 500000
        else:
            learningrate = .00001
        R = .01
        xdimension = self.x.shape[1]
        neuorons = 3
        self.w = np.random.rand(xdimension + 1, neuorons)
        tempX = np.insert(self.x, 0, 1, axis=1)
        tempX = np.array(tempX, dtype=np.float64)
        validsize = int(.2 * len(self.x))
        validsetX = tempX[0:validsize, :]
        trainX = tempX[validsize:, :]
        validsetY = self.y[0:validsize]
        trainY = self.y[validsize:]
        previouserror = sys.maxint
        count = 0
        self.wprime = np.random.rand(neuorons + 1, 1)
        finalW = self.w
        finalWprime = self.wprime
        iteration = 0
        momentum = .9
        prevloss = np.random.rand(self.w.shape[0], self.w.shape[1])
        prevlossprime = np.random.rand(self.wprime.shape[0], self.wprime.shape[1])

        while True:
            u = np.dot(self.w.T, trainX.T)
            h = expit(u)
            temph = h
            h = np.insert(h, 0, 1, axis=0)
            h = np.array(h, dtype=np.float64)
            uprime = np.dot(self.wprime.T, h)
            yprime = expit(uprime)
            uvalid = np.dot(self.w.T, validsetX.T)
            hvalid = expit(uvalid)
            hvalid = np.insert(hvalid, 0, 1, axis=0)
            uvalidprime = np.dot(self.wprime.T, hvalid)
            yvalidprime = expit(uvalidprime)

            currenterror = (np.mean((validsetY - yvalidprime) ** 2)) / 2

            if iteration >= maxiteration:
                finalW = self.w
                finalWprime = self.wprime
                break

            if currenterror > previouserror:
                if count == 0:
                    finalW = self.w
                    finalWprime = self.wprime
                count = count + 1
                if count >= 10 and iteration > 100000:
                    break
            else:
                count = 0
            previouserror = currenterror
            regwprime = np.multiply(learningrate, np.multiply(2, np.multiply(R, self.wprime)))
            l2delta = np.multiply(np.subtract(yprime, trainY.T), np.multiply(yprime, np.subtract(1, yprime)))
            lossprime = np.multiply(learningrate, np.dot(l2delta, h.T))
            self.wprime = np.subtract(self.wprime, lossprime.T)
            self.wprime = np.subtract(self.wprime, regwprime)
            self.wprime = np.subtract(self.wprime, np.multiply(momentum, prevlossprime))
            prevlossprime = lossprime.T
            tempWprime = self.wprime[1:]

            regw = np.multiply(learningrate, np.multiply(2, np.multiply(R, self.w)))
            l1delta = (l2delta.T.dot(tempWprime.T)).T * (temph * (1 - temph))
            loss = learningrate * (trainX.T.dot(l1delta.T))
            self.w = np.subtract(self.w, loss)
            self.w = np.subtract(self.w, regw)
            self.w = np.subtract(self.w, np.multiply(momentum, prevloss))
            prevloss = loss
            iteration = iteration + 1
        self.w = finalW
        self.wprime = finalWprime
        self.params = [(self.w[0, :], self.w[1:, :]), (self.wprime[0], self.wprime[1:])]  # [(w,b),(w,b)]

    def get_params(self):
        """
         This code will return Weights and Bias of the trained network.

        Returns:
            tuple of numpy.ndarray: (w, b).

        """
        return self.params

    def get_predictions(self, x):
        """
        This method will return prediction for unseen data.

        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.

        Returns:
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of
                            ``x``
        """
        testX = np.insert(x, 0, 1, axis=1)
        utest = np.dot(self.w.T, testX.T)
        htest = expit(utest)
        htest = np.insert(htest, 0, 1, axis=0)
        utestprime = np.dot(self.wprime.T, htest)
        ytestprime = expit(utestprime)
        predY = ytestprime > .5
        predY = predY.astype(int)
        predY = predY.flatten()
        return predY


class mlnn(object):
    """
    This code will train and test  the Neural Network for image data.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.

    """
    def __init__(self, data, labels):
        self.x = data / 255.0
        self.y = labels
        maxiteration=40000
        if self.x.shape[0]<=100:
            learningrate = .0001
        elif self.x.shape[0]<=500:
            learningrate=.0001
        else:
            learningrate = .00001
        if  self.x.shape[0]>500:
            maxiteration=15000
        R = 0.01
        neuorons = 100
        self.w = 0.01 * np.random.rand(self.x.shape[1] + 1, neuorons)
        tempX = np.insert(self.x, 0, 1, axis=1)
        tempX = np.array(tempX, dtype=np.float64)
        validsize = int(.2 * len(self.x))
        validsetX = tempX[0:validsize, :]
        validsetX -= np.mean(validsetX, axis=0)
        trainX = tempX[validsize:, :]
        trainX -= np.mean(trainX, axis=0)
        validsetY = self.y[0:validsize]
        trainY = self.y[validsize:]
        previouserror = sys.maxint
        count = 0
        self.wprime = 0.01 * np.random.rand(neuorons + 1, 1)
        finalW = self.w
        finalWprime = self.wprime
        iteration = 0
        while True:
            randomTrainX = trainX
            randomTrainY = trainY
            h = 1.0 / (1.0 + np.exp(-1.0 * np.dot(self.w.T, randomTrainX.T)))
            temph = h
            h = np.insert(h, 0, 1, axis=0)
            uprime = np.dot(self.wprime.T, h)
            yprime = expit(uprime)
            uvalid = np.dot(self.w.T, validsetX.T)
            hvalid = expit(uvalid)
            hvalid = np.insert(hvalid, 0, 1, axis=0)
            uvalidprime = np.dot(self.wprime.T, hvalid)
            yvalidprime = expit(uvalidprime)
            currenterror = (np.mean((validsetY - yvalidprime) ** 2)) / 2

            if iteration >= maxiteration:
                finalW = self.w
                finalWprime = self.wprime
                break

            if currenterror > previouserror:
                if count == 0:
                    finalW = self.w
                    finalWprime = self.wprime
                count = count + 1
                if count >= 10 and iteration>=10000:
                    break
            else:
                count = 0

            previouserror = currenterror


            regwprime = np.multiply(learningrate, np.multiply(2, np.multiply(R, self.wprime)))
            l2delta = np.multiply(np.subtract(yprime, randomTrainY.T), np.multiply(yprime, np.subtract(1.0, yprime)))
            lossprime = np.multiply(learningrate, np.dot(l2delta, h.T))
            self.wprime = np.subtract(self.wprime, lossprime.T)
            self.wprime = np.subtract(self.wprime, regwprime)
            tempWprime = self.wprime[1:]

            regw = np.multiply(learningrate, np.multiply(2, np.multiply(R, self.w)))
            l1delta = (l2delta.T.dot(tempWprime.T)).T * (temph * (1.0 - temph))
            loss = learningrate * (randomTrainX.T.dot(l1delta.T))
            self.w = np.subtract(self.w, loss)
            self.w = np.subtract(self.w, regw)
            iteration = iteration + 1
        self.w = finalW
        self.wprime = finalWprime

        self.params = [(self.w[0, :], self.w[1:, :]), (self.wprime[0], self.wprime[1:])]  # [(w,b),(w,b)]

    def get_params(self):
        """
        This code will return Weights and Bias of the trained network.

        Returns:
            tuple of numpy.ndarray: (w, b).

        """
        return self.params

    def get_predictions(self, x):
        """
        This method will return prediction for unseen data.

        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.

        Returns:
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of
                            ``x``
        """
        x = x / 255.0
        x -= np.mean(x, axis=0)
        testX = np.insert(x, 0, 1, axis=1)
        utest = np.dot(self.w.T, testX.T)
        htest = expit(utest)
        htest = np.insert(htest, 0, 1, axis=0)
        utestprime = np.dot(self.wprime.T, htest)
        ytestprime = expit(utestprime)
        predY = ytestprime > .5
        predY = predY.astype(int)
        predY = predY.flatten()
        return predY


if __name__ == '__main__':
    pass 
