import aws
import os
import os.path

c = aws.RealAWSClient()

'''
for f in c.listObjects(aws.S3BUCKET, 'tflogs'):
   path = f['Key']
   if not os.path.isdir(os.path.dirname(path)):
       os.mkdir(os.path.dirname(path))
   c.downloadFile(aws.S3BUCKET, path, path)

#for f in c.listObjects(aws.S3BUCKET, 'tfmodels'):
#   path = f['Key']
#   c.downloadFile(aws.S3BUCKET, path, path)

for r in [50, 52, 53, 54, 55]:
    c.downloadFile(aws.S3BUCKET, 'run{}.settings'.format(r), os.path.join('tfmodels', 'run{}.settings'.format(r)))
    c.downloadFile(aws.S3BUCKET, 'tfmodels/run{}best'.format(r), os.path.join('tfmodels', 'run{}'.format(r)))
    c.downloadFile(aws.S3BUCKET, 'tfmodels/run{}best.meta'.format(r), os.path.join('tfmodels', 'run{}.meta'.format(r)))
'''

objects = c.listObjects(aws.S3BUCKET, 'goo')
for obj in objects:
   key = obj['Key']
   c.downloadFile(aws.S3BUCKET, key, os.path.join('goo', key))
