import aws
import command
import http.server
import logging
import logging.handlers
import nn
import os
import os.path
import pickle
import time

def setupLogging(awsClient):
    logFile = awsClient.getLogFile()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    needRoll = os.path.exists(logFile)
    handler = logging.handlers.RotatingFileHandler(logFile, backupCount=10)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(handler)
    if needRoll:
        handler.doRollover()

class WorkerServer(http.server.BaseHTTPRequestHandler):
    def __init__(self, logFile, *args):
        self.logFile = logFile
        super().__init__(*args)

    def do_GET(self):
        try:
            with open(self.logFile, 'r') as fd:
                self.send_response(200)
                self.send_header("Content-type", "text/html")
                self.end_headers()
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
    logging.info('attempting port %d', port)
    server = http.server.HTTPServer(('', port), createServer)
    logging.info('listening on %d', port)
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
            self.cmdClient.sendHeartBeatRequest(workId)

def main():
    awsClient = aws.RealAWSClient()
    setupLogging(awsClient)
    awsClient.downloadFile(aws.S3BUCKET, "testData.csv", "testData.csv")

    if not os.path.exists('tfmodels'):
        os.mkdir('tfmodels')
    if not os.path.exists('tflogs'):
        os.mkdir('tflogs')

    t = threading.Thread(target=logServerThread, args=(awsClient,))
    t.start()

    cmdClient = command.CommandClient(awsClient.getCommandInstance()['PrivateIpAddress'],
                                      awsClient.getInstanceId())

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
        mad, mse = nn.nn(settings, lambda: h.maybeSendHeartBeat())

        logging.info('Uploading results')

        awsClient.putObject(aws.S3BUCKET, 'run{}.result'.format(workId), pickle.dumps({'mad': mad, 'mse': mse}))
        awsClient.uploadFile(os.path.join('tfmodels', 'run{}.meta'.format(workId)), aws.S3BUCKET, 'tfmodels/run{}.meta'.format(workId))
        awsClient.uploadFile(os.path.join('tfmodels', 'run{}'.format(workId)), aws.S3BUCKET, 'tfmodels/run{}'.format(workId))
        for f in os.listdir(os.path.join('tflogs/runtrain{}'.format(workId))):
            awsClient.uploadFile(os.path.join('tflogs', 'runtrain{}'.format(workId), f), aws.S3BUCKET, 'tflogs/runtrain{}/{}'.format(workId, f))
        for f in os.listdir(os.path.join('tflogs/runvalid{}'.format(workId))):
            awsClient.uploadFile(os.path.join('tflogs', 'runvalid{}'.format(workId), f), aws.S3BUCKET, 'tflogs/runvalid{}/{}'.format(workId, f))

        logging.info('Done uploading results')

        cmdClient.sendFinishedWorkRequest(workId)

if __name__ == "__main__":
    try:
        main()
    except:
        logging.error('Uncaught exception', exc_info=True)
    os._exit(0)
