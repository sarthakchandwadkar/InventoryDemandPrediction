from azureml.core import Workspace, Model
ws = Workspace.from_config()

model = Model.register(workspace=ws,
                       model_path="inventory_demand_model.pkl",
                       model_name="InventoryDemandModel")
