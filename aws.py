import base64
import boto3
import io
import logging
import os
import requests
import subprocess

S3BUCKET = 'timeserieslearning'

class RealAWSClient:
    def __init__(self):
        self.ec2 = boto3.client('ec2', 'us-west-2')
        self.s3 = boto3.client('s3', 'us-west-2')

    def getObject(self, bucket, key, byteRange=None):
        try:
            if byteRange is not None:
                (startOffset, endOffset) = byteRange
                byteRange = 'bytes={}-{}'.format(startOffset, endOffset)
                return self.s3.get_object(Bucket=bucket, Key=key, Range=byteRange)
            else:
                return self.s3.get_object(Bucket=bucket, Key=key)
        except:
            logging.error('get_object failed', exc_info=True)
            return None

    def getObjectBody(self, bucket, key, byteRange=None):
        obj = self.getObject(bucket, key, byteRange)
        if obj is None:
            return None
        return obj['Body']

    def putObject(self, bucket, key, value):
        try:
            self.s3.put_object(Bucket=bucket, Key=key, Body=value)
            return True
        except:
            logging.error('put_object failed', exc_info=True)
            return False

    def uploadFile(self, fileName, bucket, key):
        try:
            self.s3.upload_file(fileName, bucket, key)
            return True
        except:
            logging.error('upload_file failed', exc_info=True)
            return False

    def downloadFile(self, bucket, key, fileName):
        try:
            self.s3.download_file(bucket, key, fileName)
            return True
        except:
            logging.error('upload_file failed', exc_info=True)
            return False

    def listObjects(self, bucket, prefix=None):
        try:
            if prefix is not None:
                d = self.s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            else:
                d = self.s3.list_objects_v2(Bucket=bucket)
            if 'Contents' in d:
                return d['Contents']
            else:
                return []
        except:
            logging.error('list_objects_v2 failed', exc_info=True)
            return None

    def getSpotFleetRequests(self):
        try:
            return self.ec2.describe_spot_fleet_requests()['SpotFleetRequestConfigs']
        except:
            logging.error('describe_spot_fleet_requests failed', exc_info=True)
            return None

    def cancelSpotFleetRequest(self, spotFleetRequestId, terminateInstances):
        try:
            self.ec2.cancel_spot_fleet_requests(SpotFleetRequestIds=[spotFleetRequestId],
                                                TerminateInstances=terminateInstances)
            return True
        except:
            logging.error('cancel_spot_fleet_requests failed', exc_info=True)
            return False

    def modifySpotFleetRequest(self, spotFleetRequestId, vcpus):
        try:
            self.ec2.modify_spot_fleet_request(SpotFleetRequestId=spotFleetRequestId,
                                              TargetCapacity=vcpus,
                                              ExcessCapacityTerminationPolicy='default')
            return True
        except:
            logging.error('modify_spot_fleet_requests failed', exc_info=True)
            return False

    def sendSpotFleetRequest(self, pricePerHour, capacity):
        if pricePerHour * capacity > 10:
            logging.error('Tried to spend more than $10/hr')
            return None

        net1 = {'NetworkInterfaces': [
                            {
                                "AssociatePublicIpAddress": True,
                                "SubnetId": "subnet-659af601",
                                "SecondaryPrivateIpAddressCount": 0,
                                "DeviceIndex": 0,
                                "DeleteOnTermination": True,
                                "Groups": ["sg-fc6af29a"]
                            }
                        ],}
        net2 = {'NetworkInterfaces': [
                            {
                                "AssociatePublicIpAddress": True,
                                "SubnetId": "subnet-581bd500",
                                "SecondaryPrivateIpAddressCount": 0,
                                "DeviceIndex": 0,
                                "DeleteOnTermination": True,
                                "Groups": ["sg-fc6af29a"]
                            }
                        ],}
        net3 = {'NetworkInterfaces': [
                            {
                                "AssociatePublicIpAddress": True,
                                "SubnetId": "subnet-980f88ee",
                                "SecondaryPrivateIpAddressCount": 0,
                                "DeviceIndex": 0,
                                "DeleteOnTermination": True,
                                "Groups": ["sg-fc6af29a"]
                            }
                        ],}
        types = [('c4.xlarge', 4), ('c4.2xlarge', 8), ('c4.4xlarge', 16), ('c4.8xlarge', 32), ('c3.xlarge', 4), ('c3.2xlarge', 8), ('c3.4xlarge', 16), ('c3.8xlarge', 32)]
        typesDict = [{'InstanceType': name, 'WeightedCapacity': cap} for (name, cap) in types]
        nets = [net1, net2, net3]
        with open('/home/ec2-user/source/init_worker', 'r') as fd:
            userData = fd.read()
        launchSpec = {
                        'ImageId': 'ami-b34886d3',
                        'KeyName': 'tomconerly',
                        'UserData': base64.b64encode(bytes(userData, 'utf-8')).decode('utf-8'),
                        'IamInstanceProfile': {'Arn': 'arn:aws:iam::328628590430:instance-profile/EC2'},
                        'EbsOptimized': False,
                        'BlockDeviceMappings': [
                                {
                                  'DeviceName': '/dev/xvda',
                                  'Ebs': {
                                    'DeleteOnTermination': True,
                                    'VolumeType': 'gp2',
                                    'VolumeSize': 8,
                                    'SnapshotId': 'snap-296981d4'
                                  }
                                }
                              ],
                        }
        launchSpecs = [{**launchSpec, **n, **t} for n in nets for t in typesDict]
        try:
            # TODO do I need  'ClientToken': self.getMyInstance()['ClientToken'],
            config = {'SpotPrice': str(pricePerHour),
                      'TargetCapacity': capacity,
                      'IamFleetRole': 'arn:aws:iam::328628590430:role/aws-ec2-spot-fleet-role',
                      'ExcessCapacityTerminationPolicy': 'default',
                      'AllocationStrategy': 'lowestPrice',
                      'Type': 'maintain',
                      'LaunchSpecifications': launchSpecs
                     }
            r = self.ec2.request_spot_fleet(SpotFleetRequestConfig=config)
            return r['SpotFleetRequestId']
        except:
            logging.error('Spot fleet request failed', exc_info=True)
            return None

    def getSpotFleetInstances(self, spotFleetRequestId):
        try:
            return self.ec2.describe_spot_fleet_instances(SpotFleetRequestId=spotFleetRequestId)['ActiveInstances']
        except:
            logging.error('describe_spot_fleet_instances failed', exc_info=True)
            return None

    def getInstanceId(self):
        return requests.get('http://169.254.169.254/latest/meta-data/instance-id').content.decode()

    def getInstanceType(self):
        return requests.get('http://169.254.169.254/latest/meta-data/instance-type').content.decode()

    def getMyInstance(self):
        inst = self.ec2.describe_instances(InstanceIds=[self.getInstanceId()])
        if len(inst['Reservations']) != 1 or len(inst['Reservations'][0]['Instances']) != 1:
            raise ValueError('invalid response')
        return inst['Reservations'][0]['Instances'][0]

    def getInstances(self, instanceIds):
        res = self.ec2.describe_instances(InstanceIds=instanceIds)
        return sum([r['Instances'] for r in res['Reservations']], [])

    def getCommandInstance(self):
        inst = self.ec2.describe_instances(Filters=[{'Name': 'tag:role', 'Values': ['command']}])
        if len(inst['Reservations']) != 1 or len(inst['Reservations'][0]['Instances']) != 1:
            raise ValueError('invalid response')
        return inst['Reservations'][0]['Instances'][0]

    def terminateSelf(self):
        try:
            self.ec2.terminate_instances(InstanceIds=InstanceIds[getMyInstance()])
            return True
        except:
            logging.error('terminate_instances failed', exc_info=True)
            return False

    def checkPendingTermination(self):
        r = requests.get('http://169.254.169.254/latest/meta-data/spot/termination-time')
        if r.status_code != 200:
            return False
        terminationTime = dateutil.parser.parse(r.content.decode())
        unixTerminationTime = terminationTime.timeStamp()
        # Say we are getting terminated if termination time is in the future or 2 minutes in the past
        return unixTerminationTime + 120 > time.time()

    def cpuCount(self):
        return os.cpu_count()

    def onAWS(self):
        return True

    def getLogFile(self):
        return '/home/ec2-user/log'
