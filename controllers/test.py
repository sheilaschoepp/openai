import os
import optuna

# Fetch the database URL from the environment variable
database_url = os.environ.get("SAC_OPTUNA_DB_URL")

# Create an Optuna study
study = optuna.create_study(
    study_name="example_study",
    storage=database_url,
    load_if_exists=True,
    direction="minimize",
)

# Add a dummy trial
def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2

study.optimize(objective, n_trials=1)