import aws
import command
import http.server
import logging
import logging.handlers
import nn
from nn import Settings, StepScore
import os
import os.path
import pickle
import threading
import time

class WorkerServer(http.server.BaseHTTPRequestHandler):
    def __init__(self, logFile, *args):
        self.logFile = logFile
        super().__init__(*args)

    def do_GET(self):
        try:
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            with open(self.logFile, 'r') as fd:
                self.wfile.write(bytes(fd.read().replace('\n', '\n<br>'), 'utf-8'))
        except:
            logging.error('Uncaught exception in get', exc_info=True)

    def log_message(self, format, *args):
        logging.info(format, *args)

    def log_error(self, format, *args):
        logging.error(format, *args)

def logServerThread(awsClient):
    logging.info('log server thread')
    port = 80
    def createServer(*args):
        return WorkerServer(awsClient.getLogFile(), *args)
    server = http.server.HTTPServer(('', port), createServer)
    server.serve_forever()
    server.server_close()

class HeartBeat:
    def __init__(self, cmdClient, workId):
        self.cmdClient = cmdClient
        self.workId = workId
        self.last = time.time()

    def maybeSendHeartBeat(self):
        t = time.time()
        if t - self.last > 60:
            self.last = t
            self.cmdClient.sendHeartBeatRequest(self.workId)

def main():
    awsClient = aws.RealAWSClient()
    command.setupLogging(awsClient)
    awsClient.downloadFile(aws.S3BUCKET, "training.csv", "training.csv")

    if not os.path.exists('tfmodels'):
        os.mkdir('tfmodels')
    if not os.path.exists('tflogs'):
        os.mkdir('tflogs')

    t = threading.Thread(target=logServerThread, args=(awsClient,))
    t.start()

    cmdClient = command.CommandClient(awsClient.getCommandInstance()['PrivateIpAddress'],
                                      awsClient.getInstanceId())
    myType = awsClient.getInstanceType()

    while True:
        workId = cmdClient.sendGetWorkRequest()
        if workId == -1:
            logging.info('WorkId -1, but told to wait around. Sleeping')
            time.sleep(60)
            continue

        if workId == None:
            logging.info('No work to do, terminating myself')
            awsClient.terminateSelf()
            return

        h = HeartBeat(cmdClient, workId)

        settings = pickle.loads(awsClient.getObjectBody(aws.S3BUCKET, 'run{}.settings'.format(workId)).read())
        if settings.resumeRun is not None:
            name = 'tfmodels/run{}.meta'.format(settings.resumeRun)
            awsClient.downloadFile(aws.S3BUCKET, name, name)
            name = 'tfmodels/run{}'.format(settings.resumeRun)
            awsClient.downloadFile(aws.S3BUCKET, name, name)
        if myType in ['c3.2xlarge', 'c4.2xlarge']:
            settings.trainingTime /= 2
        elif myType in ['c3.4xlarge', 'c4.4xlarge']:
            settings.trainingTime /= 4
        elif myType in ['c3.8xlarge', 'c4.8xlarge']:
            settings.trainingTime /= 8

        history = nn.nn(settings, lambda: h.maybeSendHeartBeat())

        logging.info('Uploading results')

        awsClient.putObject(aws.S3BUCKET, 'run{}.result'.format(workId), pickle.dumps(history))
        awsClient.uploadFile(os.path.join('tfmodels', 'run{}.meta'.format(workId)), aws.S3BUCKET, 'tfmodels/run{}.meta'.format(workId))
        awsClient.uploadFile(os.path.join('tfmodels', 'run{}'.format(workId)), aws.S3BUCKET, 'tfmodels/run{}'.format(workId))
        awsClient.uploadFile(os.path.join('tfmodels', 'run{}best.meta'.format(workId)), aws.S3BUCKET, 'tfmodels/run{}best.meta'.format(workId))
        awsClient.uploadFile(os.path.join('tfmodels', 'run{}best'.format(workId)), aws.S3BUCKET, 'tfmodels/run{}best'.format(workId))

        logging.info('Done uploading results')

        cmdClient.sendFinishedWorkRequest(workId)

if __name__ == "__main__":
    try:
        main()
    except:
        logging.error('Uncaught exception', exc_info=True)
    os._exit(0)
