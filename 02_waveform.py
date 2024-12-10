#!/usr/bin/env python
# Created by "Thieu" at 21:33, 04/11/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from graforvfl import DataTransformer, GfoRvflTuner
from mealpy import IntegerVar, StringVar, FloatVar
from data_util import get_waveform
from config import Config


# Function to train, test, and evaluate a model for a single seed
def run_trial(model, seed, data, param_bounds, cf):
    X_train, X_test, y_train, y_test = data

    # Initialize model
    tuner = GfoRvflTuner(problem_type=cf.PROBLEM_CLS, bounds=param_bounds, cv=cf.CV, scoring=cf.LOSS_CLS,
                         optimizer=model["class"], optimizer_paras=model["paras"], verbose=cf.VERBOSE, seed=seed)
    # Train the model
    tuner.fit(X=X_train, y=y_train)

    # Collect epoch-wise training loss
    res_epoch_loss = [{"model_name": model["name"], "seed": seed, "epoch": epoch + 1, "loss": loss}
                      for epoch, loss in enumerate(tuner.loss_train)]

    # Predict and evaluate
    y_pred = tuner.predict(X_test)
    res = tuner.best_estimator.evaluate(y_test, y_pred, list_metrics=cf.LIST_METRIC_CLS)
    print(tuner.best_estimator)
    print(tuner.best_params)
    res_predict = {"model_name": model["name"], "seed": seed, **res}

    # Best set of parameter
    res_params = {"model_name": model['name'], "seed": seed, **tuner.best_params}
    return res_epoch_loss, res_predict, res_params

if __name__ == "__main__":
    ## Load data object
    # 5000 samples, 40 features, 3 classes
    X_train, X_test, y_train, y_test = get_waveform()

    ## Scaling dataset
    dt = DataTransformer(scaling_methods=("minmax",))
    X_train_scaled = dt.fit_transform(X_train)
    X_test_scaled = dt.transform(X_test)

    data = (X_train_scaled, X_test_scaled, y_train, y_test)
    Path(f"{Config.PATH_SAVE}/{Config.DATA_02}").mkdir(parents=True, exist_ok=True)

    # Design the boundary (parameters)
    PARAM_BOUNDS = [
        IntegerVar(lb=5, ub=150, name="size_hidden"),
        StringVar(valid_sets=("none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu",
                              "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid", "silu",
                              "swish", "mish", "hard_shrink"), name="act_name"),
        StringVar(valid_sets=("orthogonal", "he_uniform", "he_normal", "glorot_uniform", "glorot_normal",
                              "lecun_uniform", "lecun_normal", "random_uniform", "random_normal"),
                  name="weight_initializer"),
        StringVar(valid_sets=("MPI", "L2",), name="trainer"),
        FloatVar(lb=0.01, ub=50., name="alpha")
    ]

    # Run trials in parallel for all models and seeds
    all_epoch_losses = []
    all_results = []
    all_best_params = []  # To store the best parameters for each trial

    with ProcessPoolExecutor(max_workers=Config.N_WORKERS) as executor:
        futures = []
        for model in Config.LIST_MODELS:
            for seed in Config.LIST_SEEDS:
                futures.append(executor.submit(run_trial, model, seed, data, PARAM_BOUNDS, Config))

        # Collect results as they complete
        for future in as_completed(futures):
            res_epoch_loss, res_predict, res_params = future.result()
            all_epoch_losses.extend(res_epoch_loss)  # Add all epoch-wise losses for this trial
            all_results.append(res_predict)  # Add evaluation result for this trial
            all_best_params.append(res_params)  # Save best parameters for this trial

    # Create DataFrames with headers
    df_loss = pd.DataFrame(all_epoch_losses)  # Each row is a single epoch loss for a model/seed
    df_result = pd.DataFrame(all_results)  # Each row is a summary of metrics for a model/seed
    df_best_params = pd.DataFrame(all_best_params)  # Each row is the best hyperparameters for a trial

    # Save DataFrames to CSV with headers
    df_loss.to_csv(f"{Config.PATH_SAVE}/{Config.DATA_02}/{Config.RESULT_LOSS}", index=False, header=True)
    df_result.to_csv(f"{Config.PATH_SAVE}/{Config.DATA_02}/{Config.RESULT_METRICS}", index=False, header=True)
    df_best_params.to_csv(f"{Config.PATH_SAVE}/{Config.DATA_02}/{Config.RESULTS_PARAMS}", index=False, header=True)

    print(f"Done with data: {Config.DATA_02}.")
