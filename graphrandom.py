import collections
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

prefix = 'goc'

StepScore = collections.namedtuple('StepScore', ['trainMAD', 'trainMSE', 'validMAD', 'validMSE', 'step'])

class Settings:
    def __init__(self, randomArgs, args):
        self.runId = args.runId
        self.stopAfterNoImprovement = args.stopAfterNoImprovement
        self.trainingTime = args.trainingTime
        self.validateInterval = args.validateInterval

        if randomArgs:
            self.resumeRun = None
            self.trainingPercent = 0.8
            self.validationOffset = 0.8

            numHiddenLayers = random.randint(1, 6)
            self.hiddenLayerSizes = [random.choice([10, 20, 30, 40, 60, 80, 100, 140, 180]) for i in range(numHiddenLayers)]
            self.batchSize = random.choice([1, 2, 4, 8, 16, 32, 64, 128, 256])
            self.dropout = random.choice([1, 1, 1, random.uniform(0.3, 0.9)])
            self.normalizeInput = random.choice([False, True])
            self.ordinalNan = random.choice([False, True])
            self.learningRate0 = random.expovariate(1/.001)
            self.learningRate1 = random.expovariate(1/.0001)
            self.learningRatet = random.choice([1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7])
            self.l1reg = random.choice([0, random.expovariate(1)])
            self.l2reg = random.choice([0, random.expovariate(1)])
            self.activation = random.choice(['relu', 'relu', 'relu', 'sigmoid', 'tanh'])
            self.reshuffle = random.choice([False, True, True])
            self.nanToMean = random.choice([False, True])
            self.splitExtraLayer = random.choice([False, True])
            self.batchNorm = random.choice([False, True])
            self.clipNorm = random.choice([0, random.expovariate(1/.1)])

            for col in CATEGORICAL_COLS:
                if col == 'SUBJID':
                    setattr(self, col, random.randint(5, 25))
                else:
                    setattr(self, col, random.choice([-1, -1, -1, random.randint(5, 25)]))

        else:
            self.runId = args.runId
            self.resumeRun = args.resumeRun
            self.hiddenLayerSizes = args.hiddenLayerSizes
            self.batchSize = args.batchSize
            self.dropout = args.dropout
            self.trainingPercent = args.trainingPercent
            self.normalizeInput = args.normalizeInput
            self.validationOffset = args.validationOffset
            self.ordinalNan = args.ordinalNan
            self.learningRate0 = args.learningRate0
            self.learningRate1 = args.learningRate1
            self.learningRatet = args.learningRatet
            self.l1reg = args.l1reg
            self.l2reg = args.l2reg
            self.activation = args.activation
            self.reshuffle = args.reshuffle
            self.nanToMean = args.nanToMean
            self.splitExtraLayer = args.splitExtraLayer
            self.batchNorm = args.batchNorm
            self.clipNorm = args.clipNorm

            for col in CATEGORICAL_COLS:
                setattr(self, col, getattr(args, col))

    def compatible(self, s):
        if self.hiddenLayerSizes != s.hiddenLayerSizes:
            logging.info('Hidden layer sizes incompatible')
            return False
        if self.ordinalNan != s.ordinalNan:
            logging.info('Ordinal nan incompatible')
            return False
        if self.normalizeInput != s.normalizeInput:
            logging.info('Normalize input incompatible')
            return False
        if self.activation != s.activation:
            logging.info('Activation function incompatible')
            return False
        if self.batchNorm != s.batchNorm:
            logging.info('Batch norm imcompatible')
            return False
        for col in CATEGORICAL_COLS:
            if getattr(self, col) != getattr(s, col):
                logging.info('Categorical column {} incompatible'.format(col))
                return False
        return True

    def __str__(self):
        cat = ''
        for col in CATEGORICAL_COLS:
            name = col
            if col.startswith('COVAR_NOMINAL_'):
                name = 'cvn' + col[-1]
            cat += '{}:{},'.format(name.lower(), getattr(self, col))
        return 'Run: {}, Res: {}. Graph[Hid: {}, Norm: {}, OrdNan: {}, Cat: {}, Act: {}, BN: {}, CN: {}]<br> Training[Batch: {}, Time: {}, Drop: {}, l0: {}, l1: {}, lt: {}, train: {}, valOff: {}, l1r: {}, l2r: {}]'.format(self.runId, self.resumeRun, self.hiddenLayerSizes, 'T' if self.normalizeInput else 'F', 'T' if self.ordinalNan else 'F', cat, self.activation, self.batchNorm, self.clipNorm, self.batchSize, self.trainingTime, self.dropout, self.learningRate0, self.learningRate1, self.learningRatet, self.trainingPercent, self.validationOffset, self.l1reg, self.l2reg)

prefix = 'goc'
filterBelow = .05

runs = {}
results = {}
for key in os.listdir(prefix):
    if key.endswith('settings'):
        workId = int(key[3:-9])
        runs[workId] = pickle.loads(open(os.path.join(prefix, key), 'rb').read())
    if key.endswith('result'):
        workId = int(key[3:-7])
        results[workId] = pickle.loads(open(os.path.join(prefix, key), 'rb').read())

comb = []
for i in range(300):
    if i in runs and i in results:
        comb.append((min([s.validMAD for s in results[i]]), runs[i]))

comb.sort()
for (v, s) in comb[:20]:
    print('{:.6f} {}'.format(v, s))

def graph(name):
    x = [getattr(s, name) for (v, s) in comb if v < filterBelow]
    y = [v for (v, s) in comb if v < filterBelow]
    plt.scatter(x, y)
    plt.title(name)
    plt.show()

def hist(name, values):
    rows = int(math.sqrt(len(values)))
    cols = math.ceil(len(values) / rows)
    for i in range(len(values)):
        plt.subplot(rows, cols, i+1)
        if values[i] is not None:
            filtered = [v for (v, s) in comb if getattr(s, name) == values[i] and v < filterBelow]
        else:
            filtered = [v for (v, s) in comb if getattr(s, name) not in values and v < filterBelow]
        plt.hist(filtered, 50)
        if len(filtered) == 0:
            continue
        plt.title('{}: {:.5f} {:.5f}'.format(values[i], min(filtered), sum(filtered)/len(filtered)))
        print('Best {}: {:5f} {:5f}'.format(values[i], min(filtered), sum(filtered)/len(filtered)))
    plt.suptitle(name)
    plt.show()


def histOnOff(name, offValue):
    plt.subplot(1, 2, 1)
    filtered = [v for (v, s) in comb if getattr(s, name) == offValue and v < filterBelow]
    plt.hist(filtered, 50)
    plt.title('off: {:.5f} {:.5f}'.format(min(filtered), sum(filtered)/len(filtered)))
    print('Best off: {:5f} {:5f}'.format(min(filtered), sum(filtered)/len(filtered)))

    plt.subplot(1, 2, 2)
    filtered = [v for (v, s) in comb if getattr(s, name) != offValue and v < filterBelow]
    plt.hist(filtered, 50)
    plt.title('on: {:.5f} {:.5f}'.format(min(filtered), sum(filtered)/len(filtered)))
    print('Best on: {:5f} {:5f}'.format(min(filtered), sum(filtered)/len(filtered)))
    plt.suptitle(name)
    plt.show()

def numHiddenNodes():
    x = [sum(s.hiddenLayerSizes) for (v, s) in comb if v < filterBelow]
    y = [v for (v, s) in comb if v < filterBelow]
    plt.scatter(x, y)
    plt.title('Num Hidden Nodes')
    plt.show()

def numHiddenLayers():
    l = [(len(s.hiddenLayerSizes), v) for (v, s) in comb if v < filterBelow]
    for i in range(2, 9):
        plt.subplot(3, 3, i)
        filtered = [v for (le, v) in l if le == i and v < filterBelow]
        plt.hist(filtered, 50)
        if len(filtered) == 0:
            continue
        plt.title('{}: {:.5f} {:.5f}'.format(i, min(filtered), sum(filtered)/len(filtered)))
    plt.suptitle('Num hidden layers')
    plt.show()

def numSmallestLayer():
    x = [min(s.hiddenLayerSizes) for (v, s) in comb if v < filterBelow]
    y = [v for (v, s) in comb if v < filterBelow]
    plt.scatter(x, y)
    plt.title('smallest layer')
    plt.show()

def avgHiddenLayer():
    x = [sum(s.hiddenLayerSizes) / len(s.hiddenLayerSizes) for (v, s) in comb if v < filterBelow]
    y = [v for (v, s) in comb if v < filterBelow]
    plt.scatter(x, y)
    plt.title('avg hidden layer')
    plt.show()

def showBinary():
    hist('normalizeInput', [False, True])
    hist('ordinalNan', [False, True])
    hist('reshuffle', [False, True])
    hist('nanToMean', [False, True])
    hist('splitExtraLayer', [False, True])
    hist('batchNorm', [False, True])

def showReg():
    histOnOff('l1reg', 0)
    graph('l1reg')
    histOnOff('l2reg', 0)
    graph('l2reg')

hist('outputBias', [False, True])
graph('initialBias')
graph('weightMax')
hist('COMBINED_NOMINAL', [-1, 0, None])
hist('COMBINED_ID', [-1, 0, None])

showBinary()
showReg()

hist('batchSize', [96, 128, 192, 256, 384, 512])
histOnOff('clipNorm', 0)
graph('clipNorm')

numHiddenNodes()
numHiddenLayers()
numSmallestLayer()
avgHiddenLayer()

graph('SUBJID')
graph('learningRate0')
graph('learningRate1')

histOnOff('COVAR_NOMINAL_1', -1)
histOnOff('COVAR_NOMINAL_2', -1)
histOnOff('COVAR_NOMINAL_3', -1)
histOnOff('COVAR_NOMINAL_4', -1)
histOnOff('COVAR_NOMINAL_5', -1)
histOnOff('COVAR_NOMINAL_6', -1)
histOnOff('COVAR_NOMINAL_7', -1)
histOnOff('COVAR_NOMINAL_8', -1)
