import matplotlib
import matplotlib.pyplot
import pandas as pd

train = pd.read_csv('training.csv')
test = pd.read_csv('testData.csv')

cols = train.columns

printTopN = False
printMissingProb = False

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
            if trainProb < .01:
                break
            print('{}: {} {}'.format(a, trainProb, testProb))
        print()

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

#matplotlib.pyplot.hist2d(x, y, bins=[15, 10], range=[[0, 15], [0, 10]])
#matplotlib.pyplot.show()
