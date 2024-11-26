from azureml.core import Environment
from azureml.core.environment import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core import Model, Workspace
from azureml.core.webservice import AciWebservice

# Load workspace from the configuration file
ws = Workspace.from_config()

# Register the model
model = Model.register(model_path="inventory_demand_model.pkl",
                       model_name="InventoryDemandModelFinal",
                       workspace=ws)

# # Create a new environment for the model (Optional, but recommended)
# env = Environment(name="myenv")
# conda_dep = CondaDependencies.create(pip_packages=["scikit-learn", "numpy", "joblib", "azureml-defaults"])
# env.python.conda_dependencies = conda_dep
#
#
# # Create the inference configuration
# inference_config = InferenceConfig(entry_script="score.py", environment=env)
#
# # Define the deployment configuration
# aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
#
# # Deploy the model as a web service
# service = Model.deploy(workspace=ws,
#                        name="inventory-demand-service-v2",  # Name of the web service
#                        models=[model],
#                        inference_config=inference_config,
#                        deployment_config=aci_config)
#
# # Wait for deployment to finish
# service.wait_for_deployment(show_output=True)
#
# # Print the state of the service after deployment
# print(service.state)  # 'Healthy' if the deployment is successful
#
# # If service failed, get logs for troubleshooting
# if service.state != 'Healthy':
#     print(service.get_logs())


# from azureml.core import Environment
# from azureml.core.model import InferenceConfig
#
# # Define the environment
env = Environment(name="myenv")
# conda_dep = CondaDependencies.create(pip_packages=["scikit-learn", "numpy", "joblib", "azureml-defaults"])
# env.python.conda_dependencies = conda_dep

# Create the inference configuration
inference_config = InferenceConfig(entry_script="score.py", environment=env)

from azureml.core.webservice import AciWebservice
from azureml.core.webservice import Webservice

# Define the ACI configuration
aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)

# Deploy the model
service = Model.deploy(workspace=ws,
                       name="inventory-demand-service8",
                       models=[model],
                       inference_config=inference_config,
                       deployment_config=aci_config)
service.wait_for_deployment(show_output=True)

print(f"Service deployed at: {service.scoring_uri}")


