# TimeSeriesLearning

nn.py has all of the machine learning code

aws.py is a simple wrapper around the amazon aws api

command.py is a simple command and control server for running jobs on EC2

worker.py is a simple server that gets jobs from the command server, then uses nn.py to run them, then writes them back to S3

Everything else is just random one off scripts. 

This was for this TopCoder match https://community.topcoder.com/longcontest/?module=ViewProblemStatement&rd=16770&pm=14363
