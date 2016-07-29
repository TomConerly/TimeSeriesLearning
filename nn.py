import argparse
import logging
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import time

def shuffleParallel(L):
    for l in L:
        np.random.seed(0)
        np.random.shuffle(l)

def loadData(fileName, shuffle):
    data = pd.read_csv(fileName)
    ordinal = data[["TIMEVAR1","TIMEVAR2","COVAR_CONTINUOUS_1","COVAR_CONTINUOUS_2","COVAR_CONTINUOUS_3","COVAR_CONTINUOUS_4","COVAR_CONTINUOUS_5","COVAR_CONTINUOUS_6","COVAR_CONTINUOUS_7","COVAR_CONTINUOUS_8","COVAR_CONTINUOUS_9","COVAR_CONTINUOUS_10","COVAR_CONTINUOUS_11","COVAR_CONTINUOUS_12","COVAR_CONTINUOUS_13","COVAR_CONTINUOUS_14","COVAR_CONTINUOUS_15","COVAR_CONTINUOUS_16","COVAR_CONTINUOUS_17","COVAR_CONTINUOUS_18","COVAR_CONTINUOUS_19","COVAR_CONTINUOUS_20","COVAR_CONTINUOUS_21","COVAR_CONTINUOUS_22","COVAR_CONTINUOUS_23","COVAR_CONTINUOUS_24","COVAR_CONTINUOUS_25","COVAR_CONTINUOUS_26","COVAR_CONTINUOUS_27","COVAR_CONTINUOUS_28","COVAR_CONTINUOUS_29","COVAR_CONTINUOUS_30","COVAR_ORDINAL_1","COVAR_ORDINAL_2","COVAR_ORDINAL_3","COVAR_ORDINAL_4","COVAR_ORDINAL_5","COVAR_ORDINAL_6","COVAR_ORDINAL_7","COVAR_ORDINAL_8"]]
    ordinal = ordinal.fillna(0)
    ategorical = data[["STUDYID","SITEID","COUNTRY","SUBJID","COVAR_NOMINAL_1","COVAR_NOMINAL_2","COVAR_NOMINAL_3","COVAR_NOMINAL_4","COVAR_NOMINAL_5","COVAR_NOMINAL_6","COVAR_NOMINAL_7","COVAR_NOMINAL_8"]]
    outputs = data[["y1","y2","y3"]]
    outputs = outputs.fillna(0)
    outputsPresent = data[["COVAR_y1_MISSING","COVAR_y2_MISSING","COVAR_y3_MISSING"]].astype(int)

    npOrdinal = np.array(ordinal).astype(np.float32)
    npOutputs = np.array(outputs).astype(np.float32)
    npOutputsPresent = np.ones(outputsPresent.shape) - np.array(outputsPresent).astype(np.float32)
    npOutputsPresent = npOutputsPresent / npOutputsPresent.sum(axis=1, keepdims=True)

    if shuffle:
        shuffleParallel([npOrdinal, npOutputs, npOutputsPresent])

    return (npOrdinal, npOutputs, npOutputsPresent)

def buildGraph():
    pass

def predict(runId):
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join('tfmodels', 'run{}'.format(runId)))

def nn(runId, resumeRun, firstLayerSize, secondLayerSize, batchSize, trainingTime, dropout):
    npTrainOrdinal, npTrainOutputs, npTrainOutputsPresent = loadData('training.csv', shuffle=True)
    trainValidationBoundary = int(npTrainOutputs.shape[0] * 0.8)

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    ordinalInputs = tf.placeholder(tf.float32, [None, trainOrdinal.shape[1]], name='ordinalInputs')
    outputs = tf.placeholder(tf.float32, [None, 3], name='outputs')
    outputsPresent = tf.placeholder(tf.float32, [None, 3], name='outputsPresent')

    w1 = tf.Variable(tf.truncated_normal([trainOrdinal.shape[1], firstLayerSize], stddev=0.1), name="w1")
    b1 = tf.Variable(tf.constant(0.1, shape=[firstLayerSize]), name="b1")
    h1 = tf.matmul(ordinalInputs, w1) + b1
    z1 = tf.nn.relu(h1)
    z1drop = tf.nn.dropout(z1, keep_prob)

    w2 = tf.Variable(tf.truncated_normal([firstLayerSize, secondLayerSize], stddev=0.1), name="w2")
    b2 = tf.Variable(tf.constant(0.1, shape=[secondLayerSize]), name="b2")
    h2 = tf.matmul(z1drop, w2) + b2
    z2 = tf.nn.relu(h2)
    z2drop = tf.nn.dropout(z2, keep_prob)

    w3 = tf.Variable(tf.truncated_normal([secondLayerSize, 3], stddev=0.1), name="w3")
    b3 = tf.Variable(tf.constant(0.1, shape=[3]), name="b3")
    h3 = tf.matmul(z2drop, w3) + b3

    mse = tf.reduce_mean(tf.mul(tf.square(h3 - outputs), outputsPresent), name='mse')
    mad = tf.reduce_mean(tf.mul(tf.abs(h3 - outputs), outputsPresent) , name='mad')
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(mad)

    tf.scalar_summary('mse', mse)
    tf.scalar_summary('mad', mad)
    summary_op = tf.merge_all_summaries()

    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())
    summary_train_writer = tf.train.SummaryWriter(os.path.join('tflogs', 'runtrain{}'.format(runId)), sess.graph)
    summary_valid_writer = tf.train.SummaryWriter(os.path.join('tflogs', 'runvalid{}'.format(runId)), sess.graph)

    if resumeRun is not None:
        logging.info('Resuming from run {}'.format(resumeRun))
        saver.restore(sess, os.path.join('tfmodels', 'run{}'.format(resumeRun)))

    startTime = time.time()
    at = 0
    step = 0
    while time.time() - startTime < trainingTime:
        if at * batchSize >= trainValidationBoundary:
            logging.info('Starting over!')
            at = 0
        start = at * batchSize
        end = min(start + batchSize, trainValidationBoundary)
        at += 1
        feed_dict = {ordinalInputs: npTrainOrdinal[start:end], outputs: npTrainOutputs[start:end], outputsPresent: npTrainOutputsPresent[start:end], keep_prob: dropout}
        sess.run(train_step, feed_dict=feed_dict)

        if step % 100 == 0:
            feed_dict = {ordinalInputs: npTrainOrdinal[:trainValidationBoundary], outputs: npTrainOutputs[:trainValidationBoundary], outputsPresent: npTrainOutputsPresent[:trainValidationBoundary], keep_prob: 1.0}
            summary_train_writer.add_summary(sess.run(summary_op, feed_dict=feed_dict), step)
            summary_train_writer.flush()
        if step % 100 == 0:
            madScore, mseScore, summary = sess.run([mad, mse, summary_op], feed_dict={ordinalInputs: npTrainOrdinal[trainValidationBoundary:], outputs: npTrainOutputs[trainValidationBoundary:], outputsPresent: npTrainOutputsPresent[trainValidationBoundary:], keep_prob: 1.0})
            summary_valid_writer.add_summary(summary, step)
            summary_valid_writer.flush()
            logging.info('mse: {:.6f}, mad: {:.6f}'.format(mseScore, madScore))
            saver.save(sess, os.path.join('tfmodels', 'run{}'.format(runId)))
        step += 1

    npTestOrdinal, npTestOutputs, npTestOutputsPresent = loadData('testData.csv', shuffle=False)
    testPredictions = sess.run(h3, feed_dict={ordinalInputs:npTestOrdinal, outputs:npTestOutputs, outputsPresent: npTestOutputsPresent, keep_prob: 1.0})
    np.savetxt('pred.csv', testPredictions, delimiter=',', fmt='%.9f')

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.INFO)
    parser = argparse.ArgumentParser(description='ML fun')
    parser.add_argument('--runid', type=int, help='id of the run')
    parser.add_argument('--resumeRun', default=None, type=int, help='id of the run')
    parser.add_argument('--train', action='store_true', default=False, help='')
    parser.add_argument('--predict', action='store_true', default=False, help='')
    args = parser.parse_args()

    if args.train:
        nn(args.runid, args.resumeRun, firstLayerSize=2000, secondLayerSize=1000, batchSize=1000, trainingTime=60*60, dropout=1.0)
    elif args.predict:
        predict(args.runid)
    else:
        logging.info('doing nothing')

if __name__ == "__main__":
    try:
        main()
    except:
        logging.error('Uncaught exception', exc_info=True)
    os._exit(0)
