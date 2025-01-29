import os
import optuna

# Fetch the database URL from the environment variable
database_url = os.environ.get("SAC_OPTUNA_DB_URL")

# Check if the URL is loaded
if not database_url:
    raise ValueError("Database URL not found in environment. Make sure OPTUNA_DB_URL is set.")

# Name of the study to delete
study_name = "example_study"

# Delete the study
optuna.delete_study(study_name=study_name, storage=database_url)

print(f"Study '{study_name}' has been successfully deleted.")