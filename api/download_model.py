import os

## Import AzureML packages
from azureml.core import Workspace
from azureml.core import Model

workspace_name = os.environ.get('WORKSPACE', 'mlops-jonasfaber-karting')
subscription_id = os.environ.get('SUBSCRIPTION_ID', '7c50f9c3-289b-4ae0-a075-08784b3b9042')
resource_group = os.environ.get('RESOURCE_GROUP', 'NathanReserve')

ws = Workspace.get(name=workspace_name,
               subscription_id=subscription_id,
               resource_group=resource_group)

model = Model(ws, 'karting-cnn-vgg')
model.download(target_dir='api/outputs/', exist_ok=True)

print(model)