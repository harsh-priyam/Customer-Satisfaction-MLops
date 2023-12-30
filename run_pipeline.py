from pipelines.training_pipeline import train_pipeline
from zenml.client import Client 
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri

if __name__ == "__main__":

    
    # Run the pipeline
    train_pipeline(data_path="C:\\Users\\aarav\\Downloads\\Customer\\data\\olist_customers_dataset.csv")

# mlflow_ui --backend-store-uri "    C:\Users\aarav\AppData\Roaming\zenml\local_stores\4497ab39-be2d-4acc-9458-72c1f9d7a076\mlruns"

print(
        "Now run \n "
        f"    mlflow ui --backend-store-uri '{get_tracking_uri()}'\n"
        "To inspect your experiment runs within the mlflow UI.\n"
        "You can find your runs tracked within the `mlflow_example_pipeline`"
        "experiment. Here you'll also be able to compare the two runs.)"
    )