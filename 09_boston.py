#!/usr/bin/env python
# Created by "Thieu" at 21:33, 04/11/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from graforvfl import DataTransformer, GfoRvflTuner
from mealpy import IntegerVar, StringVar, FloatVar
from data_util import get_boston_housing


## Load data object
# 506 samples, 22 features
X_train, X_test, y_train, y_test = get_boston_housing()

## Scaling dataset
dt_X = DataTransformer(scaling_methods=("minmax", ))
X_train_scaled = dt_X.fit_transform(X_train)
X_test_scaled = dt_X.transform(X_test)

dt_y = DataTransformer(scaling_methods=("minmax", ))
y_train_scaled = dt_y.fit_transform(y_train.reshape(-1, 1))
y_test_scaled = dt_y.transform(y_test.reshape(-1, 1))

data = (X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled)
DATA_NAME = "boston"
EPOCH = 50
POP_SIZE = 20
LIST_SEEDS = [10, 15, 21, 24, 27, 29, 30, 35, 40, 42]
LIST_METRICS = ["MAE", "RMSE", "NNSE", "WI", "R", "KGE"]
PATH_SAVE = "history"
N_WORKERS = 10

# Design the boundary (parameters)
PARAM_BOUNDS = [
    IntegerVar(lb=5, ub=100, name="size_hidden"),
    StringVar(valid_sets=("none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu",
                          "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid", "silu",
                          "swish", "mish", "hard_shrink"), name="act_name"),
    StringVar(valid_sets=("orthogonal", "he_uniform", "he_normal", "glorot_uniform", "glorot_normal",
                          "lecun_uniform", "lecun_normal", "random_uniform", "random_normal"),
              name="weight_initializer"),
    StringVar(valid_sets=("MPI", "L2",), name="trainer"),
    FloatVar(lb=0.01, ub=10., name="alpha")
]

LIST_MODELS = [
    {"name": "BBO-RVFL", "class": "OriginalBBO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "SADE-RVFL", "class": "SADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "SHADE-RVFL", "class": "OriginalSHADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "LCO-RVFL", "class": "OriginalLCO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "INFO-RVFL", "class": "OriginalINFO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "QLE-SCA-RVFL", "class": "QleSCA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "SHIO-RVFL", "class": "OriginalSHIO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "EFO-RVFL", "class": "OriginalEFO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "A-EO-RVFL", "class": "AdaptiveEO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "RIME-RVFL", "class": "OriginalRIME", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "IM-ARO-RVFL", "class": "LARO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "HHO-RVFL", "class": "OriginalHHO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "AIW-PSO-RVFL", "class": "AIW_PSO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "CL-PSO-RVFL", "class": "CL_PSO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
]

# Function to train, test, and evaluate a model for a single seed
def run_trial(model, seed, data, param_bounds):
    X_train, X_test, y_train, y_test = data

    # Initialize model
    tuner = GfoRvflTuner(problem_type="regression", bounds=param_bounds, cv=5, scoring="MSE",
                         optimizer=model["class"], optimizer_paras=model["paras"], verbose=False, seed=42)
    # Train the model
    tuner.fit(X=X_train, y=y_train)

    # Collect epoch-wise training loss
    res_epoch_loss = [{"model_name": model["name"], "seed": seed, "epoch": epoch + 1, "loss": loss}
                      for epoch, loss in enumerate(tuner.loss_train)]

    # Predict and evaluate
    y_pred = tuner.predict(X_test)
    res = tuner.best_estimator.evaluate(y_test, y_pred, list_metrics=LIST_METRICS)
    print(tuner.best_estimator)
    print(tuner.best_params)
    res_predict = {"model_name": model["name"], "seed": seed, **res}

    # Best set of parameter
    res_params = {"model_name": model['name'], "seed": seed, **tuner.best_params}
    return res_epoch_loss, res_predict, res_params


if __name__ == "__main__":
    Path(f"{PATH_SAVE}/{DATA_NAME}").mkdir(parents=True, exist_ok=True)

    # Run trials in parallel for all models and seeds
    all_epoch_losses = []
    all_results = []
    all_best_params = []  # To store the best parameters for each trial

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = []
        for model in LIST_MODELS:
            for seed in LIST_SEEDS:
                futures.append(executor.submit(run_trial, model, seed, data, PARAM_BOUNDS))

        # Collect results as they complete
        for future in futures:
            res_epoch_loss, res_predict, res_params = future.result()
            all_epoch_losses.extend(res_epoch_loss)  # Add all epoch-wise losses for this trial
            all_results.append(res_predict)  # Add evaluation result for this trial
            all_best_params.append(res_params)  # Save best parameters for this trial

    # Create DataFrames with headers
    df_loss = pd.DataFrame(all_epoch_losses)  # Each row is a single epoch loss for a model/seed
    df_result = pd.DataFrame(all_results)  # Each row is a summary of metrics for a model/seed
    df_best_params = pd.DataFrame(all_best_params)  # Each row is the best hyperparameters for a trial

    # Save DataFrames to CSV with headers
    df_loss.to_csv(f"{PATH_SAVE}/{DATA_NAME}/df_loss.csv", index=False, header=True)
    df_result.to_csv(f"{PATH_SAVE}/{DATA_NAME}/df_result.csv", index=False, header=True)
    df_best_params.to_csv(f"{PATH_SAVE}/{DATA_NAME}/df_best_params.csv", index=False, header=True)

    print(f"Done with data: {DATA_NAME}.")
