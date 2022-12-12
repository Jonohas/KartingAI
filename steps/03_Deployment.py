import os

from utils import connectWithAzure
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core import Model

from dotenv import load_dotenv

# When you work locally, you can use a .env file to store all your environment variables.
# This line read those in.
load_dotenv()

MODEL_NAME = os.environ.get('MODEL_NAME')
LOCAL_DEPLOYMENT = os.environ.get('LOCAL_DEPLOYMENT')

def downloadLatestModel(ws):
    print('Downloading latest model')
    local_model_path = os.environ.get('LOCAL_MODEL_PATH')
    model = Model(ws, name=MODEL_NAME)
    downloaded_path = model.download(local_model_path, exist_ok=False)
    return downloaded_path

def main():
    ws = connectWithAzure()

    downloadLatestModel(ws)
    print('done downloading model')

    # environment = prepareEnv(ws)
    # service = prepareDeployment(ws, environment)

    # print('Waiting for deployment to finish...')
    # service.wait_for_deployment(show_output=True)

    # model = downloadLatestModel(ws)


if __name__ == '__main__':
    main()