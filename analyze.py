import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

train = pd.read_csv('training.csv')
test = pd.read_csv('testData.csv')

cols = train.columns

printTopN = False
printMissingProb = False
printTimes = False

if printTopN:
    for col in cols:
        trainUnique = len(train[col].unique())
        testUnique = len(test[col].unique())
        print('{} Train unique: {}, Test unique: {}'.format(col, trainUnique, testUnique))
        trainVC = train[col].value_counts(normalize=True, dropna=False)
        testVC = test[col].value_counts(normalize=True, dropna=False)
        for a in trainVC.axes[0][:10]:
            trainProb = trainVC[a] if a in trainVC else 0
            testProb = testVC[a] if a in testVC else 0
            if trainProb < .001:
                break
            print('{}: {} {}'.format(a, trainProb, testProb))
        #print()

if printMissingProb:
    for col in cols:
        print('{} {} {}'.format(col, train[col].count() / train[col].size, test[col].count() / test[col].size))


def timeMap(data):
    times = {}
    for row in data.itertuples():
        key = (row.STUDYID, row.SUBJID)
        value = row.TIMEVAR1
        if key not in times:
            times[key] = [value]
        else:
            times[key].append(value)
    return times

if printTimes:
    trainTimes = timeMap(train)
    testTimes = timeMap(test)

    x = []
    y = []
    freq = {}
    for (k, v) in trainTimes.items():
        #print(k, len(v), len(testTimes[k]) if k in testTimes else 0)
        x.append(len(v))
        y.append(len(testTimes[k]) if k in testTimes else 0)
        nk = (x[-1], y[-1])
        if nk not in freq:
            freq[nk] = 1
        else:
            freq[nk] += 1

    l = [(k, v) for (k, v) in freq.items()]
    l.sort(key=lambda a: a[1], reverse=True)
    accum = 0
    for (k, v) in l:
        accum += v / len(x)
        print(k, v / len(x), accum)

def numUniques(l):
    return len(set(l))

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))
CATEGORICAL_COLS = ["STUDYID","SITEID","COUNTRY","COVAR_NOMINAL_1","COVAR_NOMINAL_2","COVAR_NOMINAL_3","COVAR_NOMINAL_4","COVAR_NOMINAL_5","COVAR_NOMINAL_6","COVAR_NOMINAL_7","COVAR_NOMINAL_8"]
for s in powerset(CATEGORICAL_COLS):
    print(s, len(set(zip(*[np.array(train[col]) for col in s]))), *[len(train[col].unique()) for col in s])

