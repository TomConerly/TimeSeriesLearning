import collections
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

prefix = 'goe'

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
    if len(filtered) > 0:
        plt.title('off: {:.5f} {:.5f}'.format(min(filtered), sum(filtered)/len(filtered)))
        print('Best off: {:5f} {:5f}'.format(min(filtered), sum(filtered)/len(filtered)))

    plt.subplot(1, 2, 2)
    filtered = [v for (v, s) in comb if getattr(s, name) != offValue and v < filterBelow]
    plt.hist(filtered, 50)
    if len(filtered) > 0:
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
