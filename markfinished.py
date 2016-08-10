import aws
import command
import nn

awsClient = aws.RealAWSClient()
cmdClient = command.CommandClient('52.41.178.184',
                                  '')

workId = 14
cmdClient.sendFinishedWorkRequest(workId)
history = [StepScore(trainMAD=1, trainMSE=1, validMAD=1, validMSE=1, step=1)]
awsClient.putObject(aws.S3BUCKET, '{}{}.result'.format(command.prefix, workId), pickle.dumps(history))
