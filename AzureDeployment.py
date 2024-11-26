from azureml.core import Environment
from azureml.core.model import Model
from azureml.core.webservice import AciWebservice
from azureml.core import Workspace
from azureml.core.environment import CondaDependencies
from azureml.core.model import InferenceConfig

# Load the workspace from the config file
ws = Workspace.from_config()

# Create environment from the environment.yml
env = Environment.from_conda_specification(name="myenv", file_path="environment.yml")

# Load the trained model from the workspace
# Ensure the model is registered in the workspace before loading it
model = Model(ws, name="InventoryDemandModel")

# Define the inference configuration
inference_config = InferenceConfig(entry_script="score.py", environment=env)

# Define the deployment configuration (e.g., ACI)
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy the model as a web service
service = Model.deploy(workspace=ws,
                       name="inventory-demand-service-v6",
                       models=[model],  # Using the loaded model variable here
                       inference_config=inference_config,
                       deployment_config=aci_config)

# Wait for deployment to finish
service.wait_for_deployment(show_output=True)

# Check the service status
print(service.state)
