import argparse
import aws
import collections
import logging
import numpy as np
import os
import pandas as pd
import pickle
import tensorflow as tf
import time

CATEGORICAL_COLS = ["SUBJID","STUDYID","SITEID","COUNTRY","COVAR_NOMINAL_1","COVAR_NOMINAL_2","COVAR_NOMINAL_3","COVAR_NOMINAL_4","COVAR_NOMINAL_5","COVAR_NOMINAL_6","COVAR_NOMINAL_7","COVAR_NOMINAL_8"]

StepScore = collections.namedtuple('StepScore', ['trainMAD', 'trainMSE', 'validMAD', 'validMSE', 'step'])

def shuffleParallel(L):
    for l in L:
        np.random.seed(0)
        np.random.shuffle(l)

class Input:
    def __init__(self, fileName, shuffle, settings, normalizeFrom=None, ordinalNan=False):
        logging.info('Loading input {}'.format(fileName))
        data = pd.read_csv(fileName)
        ordinal = data[["TIMEVAR1","TIMEVAR2","COVAR_CONTINUOUS_1","COVAR_CONTINUOUS_2","COVAR_CONTINUOUS_3","COVAR_CONTINUOUS_4","COVAR_CONTINUOUS_5","COVAR_CONTINUOUS_6","COVAR_CONTINUOUS_7","COVAR_CONTINUOUS_8","COVAR_CONTINUOUS_9","COVAR_CONTINUOUS_10","COVAR_CONTINUOUS_11","COVAR_CONTINUOUS_12","COVAR_CONTINUOUS_13","COVAR_CONTINUOUS_14","COVAR_CONTINUOUS_15","COVAR_CONTINUOUS_16","COVAR_CONTINUOUS_17","COVAR_CONTINUOUS_18","COVAR_CONTINUOUS_19","COVAR_CONTINUOUS_20","COVAR_CONTINUOUS_21","COVAR_CONTINUOUS_22","COVAR_CONTINUOUS_23","COVAR_CONTINUOUS_24","COVAR_CONTINUOUS_25","COVAR_CONTINUOUS_26","COVAR_CONTINUOUS_27","COVAR_CONTINUOUS_28","COVAR_CONTINUOUS_29","COVAR_CONTINUOUS_30","COVAR_ORDINAL_1","COVAR_ORDINAL_2","COVAR_ORDINAL_3","COVAR_ORDINAL_4","COVAR_ORDINAL_5","COVAR_ORDINAL_6","COVAR_ORDINAL_7","COVAR_ORDINAL_8"]]

        oneHotColumns = []
        embeddingColumns = []
        embeddingSizes = []
        for col in CATEGORICAL_COLS:
            if getattr(settings, col) == 0:
                continue
            if getattr(settings, col) == -1:
                oneHotColumns.append(col)
            else:
                embeddingColumns.append(col)
                embeddingSizes.append(getattr(settings, col))

        if oneHotColumns == []:
            categoricalOneHot = np.zeros((data.shape[0], 0))
        else:
            categoricalOneHot = pd.get_dummies(data[oneHotColumns].astype('object'), dummy_na=True)
        categoricalEmbedding = data[embeddingColumns]
        categoricalEmbedding = categoricalEmbedding.apply(lambda x: x.astype('category').cat.codes)
        categoricalEmbedding = categoricalEmbedding.apply(lambda x: x.replace(-1, x.max() + 1))

        outputs = data[["y1","y2","y3"]]
        outputs = outputs.fillna(0)
        outputsPresent = data[["COVAR_y1_MISSING","COVAR_y2_MISSING","COVAR_y3_MISSING"]].astype(int)

        self.npOrdinal = np.array(ordinal.fillna(0)).astype(np.float32)
        if ordinalNan:
            self.npOrdinal = np.hstack([self.npOrdinal, np.array(1 - ordinal.notnull().astype(np.float32))])
        self.npCategoricalOneHot = np.array(categoricalOneHot).astype(np.float32)
        self.npCategoricalEmbedding = np.array(categoricalEmbedding).astype(np.int32)
        self.categoricalFeatureEmbedSizes = zip([x + 1 for x in self.npCategoricalEmbedding.max(axis=0)], embeddingSizes, embeddingColumns)
        self.npOutputs = np.array(outputs).astype(np.float32)
        self.npOutputsPresent = np.ones(outputsPresent.shape) - np.array(outputsPresent).astype(np.float32)
        self.npOutputsPresent = self.npOutputsPresent / self.npOutputsPresent.sum(axis=1, keepdims=True)

        if shuffle:
            shuffleParallel([self.npOrdinal, self.npCategoricalOneHot, self.npCategoricalEmbedding, self.npOutputs, self.npOutputsPresent])

        if normalizeFrom is not None:
            inputNormFrom = self if fileName == normalizeFrom else Input(normalizeFrom, shuffle=False, settings=settings, ordinalNan=ordinalNan)
            means = inputNormFrom.npOrdinal.mean(axis=0)
            std = inputNormFrom.npOrdinal.std(axis=0)
            self.npOrdinal = (self.npOrdinal - means) / std

    def roll(self, shift):
        self.npOrdinal = np.roll(self.npOrdinal, shift, axis=0)
        self.npCategoricalOneHot = np.roll(self.npCategoricalOneHot, shift, axis=0)
        self.npCategoricalEmbedding = np.roll(self.npCategoricalEmbedding, shift, axis=0)
        self.npOutputs = np.roll(self.npOutputs, shift, axis=0)
        self.npOutputsPresent = np.roll(self.npOutputsPresent, shift, axis=0)

class Settings:
    def __init__(self, args):
        self.runId = args.runId
        self.resumeRun = args.resumeRun
        self.hiddenLayerSizes = args.hiddenLayerSizes
        self.batchSize = args.batchSize
        self.trainingTime = args.trainingTime
        self.dropout = args.dropout
        self.trainingPercent = args.trainingPercent
        self.normalizeInput = args.normalizeInput
        self.validationOffset = args.validationOffset
        self.ordinalNan = args.ordinalNan
        self.learningRate0 = args.learningRate0
        self.learningRate1 = args.learningRate1
        self.learningRatet = args.learningRatet

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
        return 'Run: {}, Res: {}. Graph[Hid: {}, Norm: {}, OrdNan: {}, Cat: {}]<br> Training[Batch: {}, Time: {}, Drop: {}, l0: {}, l1: {}, lt: {}, train: {}, valOff: {}]'.format(self.runId, self.resumeRun, self.hiddenLayerSizes, 'T' if self.normalizeInput else 'F', 'T' if self.ordinalNan else 'F', cat, self.batchSize, self.trainingTime, self.dropout, self.learningRate0, self.learningRate1, self.learningRatet, self.trainingPercent, self.validationOffset)

class Graph:
    def __init__(self, settings, ordinalInputSize, categoricalOneHotInputSize, categoricalFeatureEmbedSizes):
        tf.reset_default_graph()
        logging.info('Building graph')
        self.keep_prob = tf.placeholder(tf.float32, name='dropoutRate')
        self.ordinalInputs = tf.placeholder(tf.float32, [None, ordinalInputSize], name='ordinalInputs')
        self.categoricalOneHotInputs = tf.placeholder(tf.float32, [None, categoricalOneHotInputSize], name='categoricalOneHotInputs')
        self.outputs = tf.placeholder(tf.float32, [None, 3], name='outputs')
        self.outputsPresent = tf.placeholder(tf.float32, [None, 3], name='outputsPresent')
        self.learningRate = tf.placeholder(tf.float32, [], name='learningRate')

        w11 = tf.Variable(tf.truncated_normal([ordinalInputSize, settings.hiddenLayerSizes[0]], stddev=0.1), name="w1ordinal")
        b1 = tf.Variable(tf.constant(0.1, shape=[settings.hiddenLayerSizes[0]]), name="b1")
        h1 = tf.matmul(self.ordinalInputs, w11) + b1
        if categoricalOneHotInputSize > 0:
            w12 = tf.Variable(tf.truncated_normal([categoricalOneHotInputSize, settings.hiddenLayerSizes[0]], stddev=0.1), name="w1categoricalOneHot")
            h1 += tf.matmul(self.categoricalOneHotInputs, w12)

        self.categoricalFeatureEmbedInputs = []
        for numClasses, embedSize, name in categoricalFeatureEmbedSizes:
            embedWeights = tf.Variable(tf.truncated_normal([numClasses, embedSize], stddev=0.1), name="embedWeights{}".format(name))
            embedInput = tf.placeholder(tf.int32, shape=[None], name="embedInput{}".format(name))
            embedOutput = tf.nn.embedding_lookup(embedWeights, embedInput, name="embedOutput{}".format(name))
            firstLayerWeights = tf.Variable(tf.truncated_normal([embedSize, settings.hiddenLayerSizes[0]], stddev=0.1), name="firstLayerEmbedWeights{}".format(name))
            h1 += tf.matmul(embedOutput, firstLayerWeights)
            self.categoricalFeatureEmbedInputs.append(embedInput)
        z1 = tf.nn.relu(h1, name="z1")
        z1drop = tf.nn.dropout(z1, self.keep_prob, name="z1drop")

        zdrops = [z1drop]
        for i in range(1, len(settings.hiddenLayerSizes)):
            w = tf.Variable(tf.truncated_normal([settings.hiddenLayerSizes[i-1], settings.hiddenLayerSizes[i]], stddev=0.1), name="w{}".format(i+1))
            b = tf.Variable(tf.constant(0.1, shape=[settings.hiddenLayerSizes[i]]), name="b{}".format(i+1))
            h = tf.matmul(zdrops[-1], w) + b
            z = tf.nn.relu(h, name="z{}".format(i+1))
            zdrop = tf.nn.dropout(z, self.keep_prob, name="zdrop{}".format(i+1))
            zdrops.append(zdrop)

        woutput = tf.Variable(tf.truncated_normal([settings.hiddenLayerSizes[-1], 3], stddev=0.1), name="w3")
        boutput = tf.Variable(tf.constant(0.1, shape=[3]), name="b3")
        self.houtput = tf.matmul(zdrops[-1], woutput) + boutput

        self.mse = tf.reduce_mean(tf.mul(tf.square(self.houtput - self.outputs), self.outputsPresent), name='mse')
        self.mad = tf.reduce_mean(tf.mul(tf.abs(self.houtput - self.outputs), self.outputsPresent) , name='mad')
        self.train_step = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.mad)

        self.gradients = tf.gradients(self.mad, tf.trainable_variables())

        logging.info('Done building graph')

def makeFeedDict(graph, input, start=None, end=None, keep_prob=1.0, learningRate=0.0):
    feed_dict = {graph.ordinalInputs: input.npOrdinal[start:end],
                 graph.categoricalOneHotInputs: input.npCategoricalOneHot[start:end],
                 graph.outputs: input.npOutputs[start:end],
                 graph.outputsPresent: input.npOutputsPresent[start:end],
                 graph.keep_prob: keep_prob,
                 graph.learningRate: learningRate}
    for i in range(len(graph.categoricalFeatureEmbedInputs)):
        feed_dict[graph.categoricalFeatureEmbedInputs[i]] = input.npCategoricalEmbedding[start:end,i]
    return feed_dict

def predict(settings):
    logging.info('Predicting')
    testInput = Input('testData.csv', shuffle=False, settings=settings, normalizeFrom='training.csv' if settings.normalizeInput else None, ordinalNan=settings.ordinalNan)
    graph = Graph(settings, testInput.npOrdinal.shape[1], testInput.npCategoricalOneHot.shape[1], testInput.categoricalFeatureEmbedSizes)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        logging.info('Restoring model')
        saver.restore(sess, os.path.join('tfmodels', 'run{}'.format(settings.runId)))

        testPredictions = sess.run(graph.houtput, feed_dict=makeFeedDict(graph, testInput))
        logging.info('Saving prediction')
        np.savetxt('pred.csv', testPredictions, delimiter=',', fmt='%.9f')

def evaluate(sess, graph, input, start=None, end=None):
    if start is None:
        start = 0
    if end is None:
        end = len(input.npOrdinal)

    length = end - start
    mad = 0
    mse = 0
    while start < end:
        s = start
        e = min(s + 10000, end)
        madScore, mseScore= sess.run([graph.mad, graph.mse], feed_dict=makeFeedDict(graph, input, start=s, end=e))
        mad += madScore * (e - s) / length
        mse += mseScore * (e - s) / length
        start = e

    return (mad, mse)

def nn(settings, callback=None):
    logging.info('Training')
    trainInput = Input('training.csv', shuffle=True, settings=settings, normalizeFrom='training.csv' if settings.normalizeInput else None, ordinalNan=settings.ordinalNan)
    trainingSize = int(trainInput.npOutputs.shape[0] * settings.trainingPercent)
    validationStart = int(trainInput.npOutputs.shape[0] * settings.validationOffset)
    trainInput.roll(trainingSize - validationStart)

    graph = Graph(settings, trainInput.npOrdinal.shape[1], trainInput.npCategoricalOneHot.shape[1], trainInput.categoricalFeatureEmbedSizes)

    saver = tf.train.Saver()
    logging.info('Starting training')
    history = []
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        if settings.resumeRun is not None:
            logging.info('Resuming from run {}'.format(settings.resumeRun))
            saver.restore(sess, os.path.join('tfmodels', 'run{}'.format(settings.resumeRun)))

        startTime = time.time()
        at = 0
        step = 0
        bestMSE = 1.0
        bestMAD = 1.0
        while time.time() - startTime < settings.trainingTime:
            if callback is not None:
                callback()
            if at * settings.batchSize >= trainingSize:
                at = 0
            start = at * settings.batchSize
            end = min(start + settings.batchSize, trainingSize)
            at += 1
            if step >= settings.learningRatet:
                learningRate = settings.learningRate1
            else:
                alpha = step / settings.learningRatet
                learningRate = (1 - alpha) * settings.learningRate0 + alpha * settings.learningRate1
            res = sess.run(graph.gradients + [graph.train_step], feed_dict=makeFeedDict(graph, trainInput, start=start, end=end, keep_prob=settings.dropout, learningRate=learningRate))
            gradients = res[:-1]

            if step % (100000 / settings.batchSize) == 0:
                trainMAD, trainMSE = evaluate(sess, graph, trainInput, end=trainingSize)
                validMAD, validMSE = evaluate(sess, graph, trainInput, start=trainingSize)
                history.append(StepScore(trainMAD=trainMAD, trainMSE=trainMSE, validMAD=validMAD, validMSE=validMSE, step=step))
                logging.info('step {}, mse: {:.6f}, mad: {:.6f}'.format(step, validMSE, validMAD))
                saver.save(sess, os.path.join('tfmodels', 'run{}'.format(settings.runId)))
                if validMAD < bestMAD:
                    saver.save(sess, os.path.join('tfmodels', 'run{}best'.format(settings.runId)))
                bestMSE = min(bestMSE, validMSE)
                bestMAD = min(bestMAD, validMAD)
            step += 1
        return history

def getSettingsPath(runId):
    return os.path.join('tfmodels', 'run{}.settings'.format(runId))

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description='ML fun')
    parser.add_argument('--runId', type=int, help='id of the run')
    parser.add_argument('--resumeRun', default=None, type=int, help='id of the run')
    parser.add_argument('--train', action='store_true', default=False, help='')
    parser.add_argument('--aws', action='store_true', default=False, help='Adds run to aws')
    parser.add_argument('--predict', action='store_true', default=False, help='')
    parser.add_argument('--hiddenLayerSizes', nargs='+', type=int)
    parser.add_argument('--batchSize', type=int, default=1000, help='batchSize')
    parser.add_argument('--trainingTime', type=int, default=60*60*5, help='trainingTime')
    parser.add_argument('--dropout', type=float, default=1.0, help='dropout')
    parser.add_argument('--learningRate0', type=float, default=1e-3, help='')
    parser.add_argument('--learningRate1', type=float, default=1e-4, help='')
    parser.add_argument('--learningRatet', type=float, default=100000, help='')
    parser.add_argument('--normalizeInput', action='store_true', default=False, help='')
    parser.add_argument('--trainingPercent', type=float, default=0.8, help='trainingPercent')
    parser.add_argument('--validationOffset', type=float, default=0.8, help='validationOffset')
    parser.add_argument('--ordinalNan', action='store_true', default=False, help='')
    parser.add_argument('--override', action='store_true', default=False, help='')
    for col in CATEGORICAL_COLS:
        parser.add_argument('--{}'.format(col), type=int, default=-1, help='')

    args = parser.parse_args()

    if args.train:
        logging.info('training')
        settings = Settings(args)
        logging.info('settings: {}', settings.__dict__)
        if os.path.isfile(getSettingsPath(args.runId)):
            logging.info('Run already exists! Exiting')
            return
        if args.resumeRun is not None:
            logging.info('Resuming from run: {}'.format(args.resumeRun))
            if not os.path.isfile(getSettingsPath(args.resumeRun)):
                logging.info("resumeRun doesn't exist. Exiting")
                return
            with open(getSettingsPath(args.resumeRun), 'rb') as f:
                prevSettings = pickle.load(f)
            if not settings.compatible(prevSettings):
                logging.info("Settings aren't compatible with previous settings. Exiting")
                return

        logging.info('writing settings')
        with open(getSettingsPath(args.runId), 'wb') as f:
            pickle.dump(settings, f)
        nn(settings)
    elif args.predict:
        logging.info('Predicting from run: {}'.format(args.runId))
        if not os.path.isfile(getSettingsPath(args.runId)):
            logging.info("Run doesn't exist. Exiting")
            return
        with open(getSettingsPath(args.runId), 'rb') as f:
            settings = pickle.load(f)
        predict(settings)
    elif args.aws:
        logging.info('adding training run to aws')
        settings = Settings(args)
        awsClient = aws.RealAWSClient()
        if not args.override and len(awsClient.listObjects(aws.S3BUCKET, 'run{}.settings'.format(settings.runId))) > 0:
            logging.info('run already exists, exiting')
            return
        if settings.resumeRun is not None:
            logging.info('Resuming from run: {}'.format(args.resumeRun))
            prevSettings = pickle.loads(awsClient.getObjectBody(aws.S3BUCKET, 'run{}.settings'.format(settings.resumeRun)).read())
            if not settings.compatible(prevSettings):
                logging.info("Settings aren't compatible with previous settings. Exiting")
                return
        awsClient.putObject(aws.S3BUCKET, 'run{}.settings'.format(settings.runId), pickle.dumps(settings))
    else:
        logging.info('doing nothing')

if __name__ == "__main__":
    try:
        main()
    except:
        logging.error('Uncaught exception', exc_info=True)
    os._exit(0)
