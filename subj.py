import pandas as pd
train = pd.read_csv('training.csv')

subjMap = {}
numrows = 0
gy1Sum = 0
gy1Count = 0
gy2Sum = 0
gy2Count = 0
gy3Sum = 0
gy3Count = 0
for row in train.itertuples():
    key = (row.STUDYID, row.SUBJID)
    if key not in subjMap:
        subjMap[key] = [row]
    else:
        subjMap[key].append(row)
    numrows += 1

    if not row.COVAR_y1_MISSING:
        gy1Sum += row.y1
        gy1Count += 1
    if not row.COVAR_y2_MISSING:
        gy2Sum += row.y2
        gy2Count += 1
    if not row.COVAR_y3_MISSING:
        gy3Sum += row.y3
        gy3Count += 1

scoreSame = 0.0
for row in train.itertuples():
    key = (row.STUDYID, row.SUBJID)
    y1Sum = 0
    y1Count = 0
    y2Sum = 0
    y2Count = 0
    y3Sum = 0
    y3Count = 0
    for e in subjMap[key]:
        if e.TIMEVAR1 == row.TIMEVAR1:
            continue
        if not e.COVAR_y1_MISSING:
            y1Sum += e.y1
            y1Count += 1
        if not e.COVAR_y2_MISSING:
            y2Sum += e.y2
            y2Count += 1
        if not e.COVAR_y3_MISSING:
            y3Sum += e.y3
            y3Count += 1
    y1pred = y1Sum / y1Count if y1Count != 0 else (gy1Sum / gy1Count)
    y2pred = y2Sum / y2Count if y2Count != 0 else (gy2Sum / gy2Count)
    y3pred = y3Sum / y3Count if y3Count != 0 else (gy3Sum / gy3Count)

    rowScore = 0
    rowEntries = 0
    if not row.COVAR_y1_MISSING:
        rowScore += abs(row.y1 - y1pred)
        rowEntries += 1
    if not row.COVAR_y2_MISSING:
        rowScore += abs(row.y2 - y2pred)
        rowEntries += 1
    if not row.COVAR_y3_MISSING:
        rowScore += abs(row.y3 - y3pred)
        rowEntries += 1
    scoreSame += rowScore / rowEntries

scoreGlobal = 0.0
for row in train.itertuples():
    y1pred = (gy1Sum / gy1Count)
    y2pred = (gy2Sum / gy2Count)
    y3pred = (gy3Sum / gy3Count)

    rowScore = 0
    rowEntries = 0
    if not row.COVAR_y1_MISSING:
        rowScore += abs(row.y1 - y1pred)
        rowEntries += 1
    if not row.COVAR_y2_MISSING:
        rowScore += abs(row.y2 - y2pred)
        rowEntries += 1
    if not row.COVAR_y3_MISSING:
        rowScore += abs(row.y3 - y3pred)
        rowEntries += 1
    scoreGlobal += rowScore / rowEntries

print('Score for average of same subject: {}'.format(10 * (1 - scoreSame / train.shape[0])))
print('Score for global average: {}'.format(10 * (1 - scoreGlobal / train.shape[0])))

