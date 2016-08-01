import aws
import os
import os.path

c = aws.RealAWSClient()

for f in c.listObjects(aws.S3BUCKET, 'tflogs'):
   path = f['Key']
   if not os.path.isdir(os.path.dirname(path)):
       os.mkdir(os.path.dirname(path))
   c.downloadFile(aws.S3BUCKET, path, path)

#for f in c.listObjects(aws.S3BUCKET, 'tfmodels'):
#   path = f['Key']
#   c.downloadFile(aws.S3BUCKET, path, path)
