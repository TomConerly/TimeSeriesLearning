import aws
import collections
import command
import nn
import pickle

Args = collections.namedtuple('Args', ['runId', 'trainingTime', 'validateInterval', 'stopAfterNoImprovement'])

for i in range(200):
    args = Args(runId=i, trainingTime=30*60, validateInterval=120, stopAfterNoImprovement=400)
    settings = nn.Settings(True, args)

    awsClient = aws.RealAWSClient()
    awsClient.putObject(aws.S3BUCKET, '{}{}.settings'.format(command.prefix, settings.runId), pickle.dumps(settings))
