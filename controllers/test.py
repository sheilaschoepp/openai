import os
import optuna

# Fetch the database URL from the environment variable
database_url = os.environ.get("SAC_OPTUNA_DB_URL")

# Check if the URL was loaded correctly
if not database_url:
    raise ValueError("Database URL not found in environment. Make sure OPTUNA_DB_URL is set.")

# Test connection by creating an Optuna study
study = optuna.create_study(
    study_name="example_study",
    storage=database_url,
    load_if_exists=True,
    direction="minimize",
)

print(f"Study '{study.study_name}' has been successfully created.")