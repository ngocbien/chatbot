import sys
import boto3 

AWS_ACCESS_KEY_ID = 'AKIAJJUGUS4CYTBXXGDQ'
AWS_SECRET_ACCESS_KEY = '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'
AWS_REGION = 'eu-west-3'
INSTANCE_ID = 'i-083800eccaa146d05'

USAGE = '''
$> source venv/bin/activate
$> python notebook_server start | stop | resize

!!! Rember to stop the instance when not used, it costs money
'''

INSTANCE_SIZE_MAP = {
    1: 'c5.large',
    2: 'c5.xlarge',
    3: 'c5.2xlarge',
    4: 'c5.4xlarge',
    5: 'c5.9xlarge',
    6: 'c5.18xlarge',
}


INSTANCE_SIZE_DESCRIPTION = {
    1: '2 core  - 4GB ram',
    2: '4 core  - 8GB ram',
    3: '8 core  - 16GB ram',
    4: '16 core - 32GB ram',
    5: '36 core - 72GB ram',
    6: '72 core - 144GB ram',
}


def get_connection():
    """
    Get an EC2 connexion
    return: ec2 boto ressource
    """
    print('Connecting to AWS ec2')
    session = boto3.Session(
        region_name=AWS_REGION, 
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )
    return session


def start(instance_id):
    """
    Start the 
    """
    session = get_connection()
    ec2 = session.resource('ec2')
    print('Looking for instance {}'.format(instance_id))
    instances = ec2.instances.filter(Filters=[{'Name': 'instance-id', 'Values': [instance_id]}, {'Name': 'instance-state-name', 'Values': ['stopped']}])
    if not len(list(instances.all())):
        print ('instance is already running')
        return
    print('Starting instance...')
    instances.start()
    waiter = session.client('ec2').get_waiter('instance_running')
    waiter.wait(InstanceIds=[instance_id])
    print('instance Started.')
    
def stop(instance_id):
    session = get_connection()
    ec2 = session.resource('ec2')
    print('Looking for instance {}'.format(instance_id))
    instances = ec2.instances.filter(Filters=[{'Name': 'instance-id', 'Values': [instance_id]}, {'Name': 'instance-state-name', 'Values': ['running']}])
    
    if not len(list(instances)):
        print('instance already stopped')
        return
    print('Stopping instance ...')
    instances.stop()
    waiter=session.client('ec2').get_waiter('instance_stopped')
    waiter.wait(InstanceIds=[instance_id])
    print('Instance stopped.')
    

def process_resize(instance_id, size):
    
    print('Resizing')
    
    stop(instance_id)
    
    session = get_connection()
    session.client('ec2').modify_instance_attribute(InstanceId=instance_id, Attribute='instanceType', Value=INSTANCE_SIZE_MAP[size])
    print ('Changing instance type to : ', INSTANCE_SIZE_MAP[size])
    start(instance_id)
    
    

def resize(instance_id):
    
    print ('Warning, this will Reboot the instance !')
    print ('Please choose the desired instance size : ')
    for key, value in INSTANCE_SIZE_DESCRIPTION.items():
        print (key, ' : ', value)
    size = "0"
    while not (size in [str(i) for i in range(1,7)]):
        size = input('Please specify desired size (1-6) or type exit : ')
        if size == 'exit':
            print ('exiting')
            return
    process_resize(instance_id, int(size))    
    
    
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Wrong Parameters')
        print(USAGE)

    else:
        if sys.argv[1] == 'start':
            start(INSTANCE_ID)
        elif sys.argv[1] == 'stop':
            stop(INSTANCE_ID)
        elif sys.argv[1] == 'resize':
            resize(INSTANCE_ID)
        else:
            print('Wrong parametes')
            print(USAGE)
