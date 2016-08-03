import aws
import constants
import csv
import enum
import http.server
import json
import logging
import logging.handlers
from nn import Settings, StepScore
import os
import pickle
import requests
import time
import urllib
import urllib.parse

GETWORK = 'GetWork'
HEARTBEAT = 'HeartBeat'
FINISHEDWORK = 'FinishedWork'

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

class CommandClient:
    def __init__(self, ip, instanceId):
        self.ip = ip
        self.instanceId = instanceId

    def _sendAPIRequest(self, data):
        try:
            r = requests.post('http://{}/api'.format(self.ip), data=data)
            logging.info('Api response has status: %d', r.status_code)
            return r
        except:
            logging.error('Api request failed', exc_info=True)
            return None

    def sendGetWorkRequest(self):
        logging.info('Sending get work request to: %s', self.ip)
        r = self._sendAPIRequest({'Type': GETWORK, 'InstanceId': self.instanceId})
        if r is None or r.status_code != 200:
            return None
        resp = json.loads(r.content.decode('utf-8'))
        if 'WorkId' not in resp:
            logging.info('No work returned')
            return None
        logging.info('Work return workId: %s', resp['WorkId'])
        return int(resp['WorkId'])

    def sendHeartBeatRequest(self, workId):
        logging.info('Sending heart beat request to: %s for workId: %d', self.ip, workId)
        r = self._sendAPIRequest({'Type': HEARTBEAT, 'InstanceId': self.instanceId, 'WorkId': str(workId)})
        return r is not None and r.status_code == 200

    def sendFinishedWorkRequest(self, workId):
        logging.info('Sending finished work request to: %s for workId: %d', self.ip, workId)
        r = self._sendAPIRequest({'Type': FINISHEDWORK, 'InstanceId': self.instanceId, 'WorkId': str(workId)})
        return r is not None and r.status_code == 200

class WorkPieceState(enum.Enum):
    unassigned = 1
    assigned = 2
    finished = 3

class WorkPiece:
    def __init__(self, state, time=0.0, settings={}, history=[]):
        self.state = state
        self.time = time
        self.settings = settings
        self.history = history

class CommandServer(http.server.BaseHTTPRequestHandler):
    def __init__(self, commander, *args):
        self.commander = commander
        super().__init__(*args)

    def do_GET(self):
        try:
            (code, content) = self.commander.httpGet(self.path)
            self.send_response(code)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes(content, 'utf-8'))
        except:
            logging.error('Uncaught exception in get', exc_info=True)

    def do_POST(self):
        try:
            logging.info('POST request path: %s', self.path)
            length = int(self.headers['content-length'])
            request = urllib.parse.parse_qs(self.rfile.read(length).decode('utf-8'))
            logging.info('POST request content: %s', json.dumps(request))
            (code, response) = self.commander.httpPost(self.path, request)
            self.send_response(code)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(bytes(json.dumps(response), 'utf-8'))
        except:
            logging.error('Uncaught exception in post', exc_info=True)

    def log_message(self, format, *args):
        logging.info(format, *args)

    def log_error(self, format, *args):
        logging.error(format, *args)

class Command:
    def __init__(self, awsClient):
        self.awsClient = awsClient
        self.workPieces = {}
        self.curSpotFleetReq = None

    def httpGet(self, path):
        if path == '/':
            workSummary = '''Total work items: {}<br>
                             Unassigned work items: {}<br>
                             Assigned work items: {}<br>
                             Finished work items: {}<br>
                             <a href="log">Log</a><br>'''
            workSummary = workSummary.format(len(self.workPieces),
                    len([s for s in self.workPieces.values() if s.state == WorkPieceState.unassigned]),
                    len([s for s in self.workPieces.values() if s.state == WorkPieceState.assigned]),
                    len([s for s in self.workPieces.values() if s.state == WorkPieceState.finished]))
            for (workId, w) in self.workPieces.items():
                logging.info('%d %s %f', workId, str(w.state), w.time)

            if self.curSpotFleetReq is None:
                serverSummary = '''No running servers or fleet request<br>'''
            else:
                servers = self.awsClient.getSpotFleetInstances(self.curSpotFleetReq)
                if len(servers) == 0:
                    serverSummary = '''No running servers but running fleet request<br>'''
                else:
                    ids = [s['InstanceId'] for s in servers]
                    longServers = self.awsClient.getInstances(ids)
                    logging.info('servers: %s, ids: %s, longServers: %s', str(servers), str(ids), str(longServers))
                    def lookupKey(d, key):
                        if key in d:
                            return d[key]
                        return 'missing'
                    serverSummary = 'Currently running servers:<br>'
                    serverSummary += ''.join(['Id: {}, Type: {}, <a href="http://{}">Log</a><br>'
                            .format(lookupKey(s, 'InstanceId'), lookupKey(s, 'InstanceType'), lookupKey(s, 'PublicIpAddress')) for s in longServers])

            start = '''<form action="start" method="get">
                           Price per vCPU/hour: <input type="number" name="price" step="any"><br>
                           Number of vCPUs: <input type="number" name="vcpus"><br>
                           <input type="checkbox" name="type" value="c3.xlarge" checked="checked">c3.xlarge<br>
                           <input type="checkbox" name="type" value="c3.2xlarge" checked="checked">c3.2xlarge<br>
                           <input type="checkbox" name="type" value="c3.4xlarge" checked="checked">c3.4xlarge<br>
                           <input type="checkbox" name="type" value="c3.8xlarge" checked="checked">c3.8xlarge<br>
                           <input type="checkbox" name="type" value="c4.xlarge" checked="checked">c4.xlarge<br>
                           <input type="checkbox" name="type" value="c4.2xlarge" checked="checked">c4.2xlarge<br>
                           <input type="checkbox" name="type" value="c4.4xlarge" checked="checked">c4.4xlarge<br>
                           <input type="checkbox" name="type" value="c4.8xlarge" checked="checked">c4.8xlarge<br>
                           <input type="submit" value="Start"><br>
                       </form>'''
            cancel = '<form action="cancel" method="get"><input type="submit" value="Cancel"></form>'

            def formatHistory(history, workId):
                if history == []:
                    return 'Not finished'
                bestMAD = min([s.validMAD for s in history])
                bestMSE = min([s.validMSE for s in history])
                return 'bmad: {:.6f}, bmse: {:.6f}, mad: {:.6f}, mse: {:.6f}, steps: {} <a href="result{}">Graph</a>'.format(bestMAD, bestMSE, history[-1].validMAD, history[-1].validMSE, history[-1].step, workId)
            joblist = '<br>'.join(['{} => {}'.format(w.settings, formatHistory(w.history, workId)) for (workId, w) in self.workPieces.items()])

            return (200, '{}<br>{}<br>{}{}<br>{}'.format(workSummary, serverSummary, start, cancel, joblist))
        elif path == '/log':
            with open(self.awsClient.getLogFile(), 'r') as fd:
                return (200, fd.read().replace('\n', '\n<br>'))
        elif path.startswith('/start?'):
            args = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)
            price = float(args['price'][0])
            vcpus = int(args['vcpus'][0])
            types = args['type']

            if self.curSpotFleetReq != None:
                logging.info('Modifying existing spot fleet request')
                self.awsClient.modifySpotFleetRequest(self.curSpotFleetReq, vcpus)
                return (200, "Modified existing request")
            else:
                logging.info('Sending spot fleet request')
                self.curSpotFleetReq = self.awsClient.sendSpotFleetRequest(price, vcpus, types)
                if self.curSpotFleetReq == None:
                    return (400, 'Request failed')
            return (200, 'Success!')
        elif path.startswith('/cancel?'):
            if self.curSpotFleetReq != None:
                self.awsClient.cancelSpotFleetRequest(self.curSpotFleetReq, terminateInstances=True)
                self.curSpotFleetReq = None
            return (200, 'Canceled')
        elif path.startswith('/result'):
            workId = int(path[7:])
            if workId not in self.workPieces or len(self.workPieces[workId].history) == 0:
                return (200, 'No result')
            return (200, constants.d3script('history{}'.format(workId)))
        elif path.startswith('/history'):
            workId = int(path[8:])
            if workId not in self.workPieces or len(self.workPieces[workId].history) == 0:
                return (404, 'No result')
            return (200, 'step,validMAD,trainMAD\n{}'.format('\n'.join(['{},{},{}'.format(h.step, h.validMAD, h.trainMAD) for h in self.workPieces[workId].history])))
        else:
            return (404, 'Unknown path: {}'.format(path))

    def httpPost(self, path, request):
        if path != '/api':
            self.send_response(404)
            return ('404', {'Response': 'Unknown post path: {}'.format(path)})

        if 'Type' not in request:
            return (400, {'Response': 'No type in request'})

        response = {'Response': 'OK'}
        if request['Type'][0] == GETWORK:
            logging.info('Got get work request')
            outstanding = False
            for workId, w in self.workPieces.items():
                if w.state == WorkPieceState.assigned:
                    outstanding = True
                if w.state != WorkPieceState.unassigned:
                    continue
                response['WorkId'] = workId
                logging.info('Responding with workId: %s', response['WorkId'])
                w.state = WorkPieceState.assigned
                w.time = time.time()
                break
            if 'WorkId' not in response:
                if outstanding:
                    logging.info('No work to give out, but there is unfinished work. Tell instance to stick around.')
                    response['WorkId'] = -1
                else:
                    logging.info('All work is done, cancel the fleet request')
                    self.awsClient.cancelSpotFleetRequest(self.curSpotFleetReq, terminateInstances=True)
                    self.curSpotFleetReq = None
        elif request['Type'][0] == HEARTBEAT:
            workId = int(request['WorkId'][0])
            logging.info('Got heart beat request workId: %s', workId)
            if workId in self.workPieces:
                logging.info('Found work item, time since last heartbeat: %f', time.time() - self.workPieces[workId].time)
                self.workPieces[workId].time = time.time()
                if self.workPieces[workId].state == WorkPieceState.unassigned:
                    self.workPieces[workId].state = WorkPieceState.assigned
        elif request['Type'][0] == FINISHEDWORK:
            workId = int(request['WorkId'][0])
            logging.info('Got finished work request workId: %s', workId)
            if workId in self.workPieces:
                self.workPieces[workId].state = WorkPieceState.finished
        else:
            logging.info('Got unknown request type %s', request['Type'][0])
            return (400, {'Response', 'Unknown type {}'.format(request['Type'][0])})

        return (200, response)

    def createServer(self, *args):
        return CommandServer(self, *args)

    def run(self):
        logging.info('Command starting')

        requests = [r for r in self.awsClient.getSpotFleetRequests()
                    if r['SpotFleetRequestState'] in ['submitted', 'active', 'cancelled_running', 'cancelled_terminating', 'modifying']]
        if len(requests) > 1:
            logging.error('Multiple outstanding fleet requests, cancel them all!')
            self.awsClient.cancelSpotFleetRequest([req['SpotFleetRequestId'] for req in requests],
                                          terminateInstances=True)
            return
        elif len(requests) == 1:
            self.curSpotFleetReq = requests[0]['SpotFleetRequestId']

        httpServer = http.server.HTTPServer(('', 80), self.createServer)
        httpServer.timeout = 60
        lastHeartBeatScan = 0
        lastWorkScan = 0
        while True:
            now = time.time()
            if now - lastHeartBeatScan > 60:
                lastHeartBeatScan = now
                for (workId, w) in self.workPieces.items():
                    if w.state == WorkPieceState.assigned and now - w.time > 600:
                        logging.info('No heart beat in %f seconds for workId %d marking unassigned',
                                     now - w.time, workId)
                        w.state = WorkPieceState.unassigned

            if now - lastWorkScan > 300:
               lastWorkScan = now
               runs = self.awsClient.listObjects(aws.S3BUCKET, 'run')
               for run in runs:
                   key = run['Key']
                   if key.endswith('settings'):
                       workId = int(key[3:-9])
                       if workId not in self.workPieces:
                           self.workPieces[workId] = WorkPiece(WorkPieceState.unassigned, settings=pickle.loads(self.awsClient.getObjectBody(aws.S3BUCKET, key).read()))
               for run in runs:
                   key = run['Key']
                   if key.endswith('result'):
                       workId = int(key[3:-7])
                       self.workPieces[workId].state = WorkPieceState.finished
                       if len(self.workPieces[workId].history) == 0:
                           self.workPieces[workId].history = pickle.loads(self.awsClient.getObjectBody(aws.S3BUCKET, key).read())

            httpServer.handle_request()

        httpServer.server_close()

def main():
    awsClient = aws.RealAWSClient()
    setupLogging(awsClient)
    cmd = Command(awsClient)
    cmd.run()

if __name__ == "__main__":
    try:
        main()
    except:
        logging.error('Uncaught exception', exc_info=True)
    os._exit(0)
