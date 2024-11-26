#!/usr/bin/env python
# Created by "Thieu" at 12:08, 26/11/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from graforvfl import GfoRvflTuner, DataTransformer, RvflClassifier
from mealpy import IntegerVar, StringVar, FloatVar
from data_util import get_digits
from pathlib import Path
import pandas as pd


## Load data object
# 1797 samples, 64 features, 10 classes
X_train, X_test, y_train, y_test = get_digits()

## Scaling dataset
dt = DataTransformer(scaling_methods=("minmax",))
X_train_scaled = dt.fit_transform(X_train)
X_test_scaled = dt.transform(X_test)

data = (X_train_scaled, X_test_scaled, y_train, y_test)
EPOCH = 50
POP_SIZE = 20
LIST_METRICS = ["AS", "PS", "RS", "F1S", "SS", "NPV"]
PATH_SAVE = "history"


def graforvfl_exp(model):
    # Design the boundary (parameters)
    time_start = time.perf_counter()
    seed = 42
    PARAM_BOUNDS = [
        IntegerVar(lb=5, ub=150, name="size_hidden"),
        StringVar(valid_sets=("none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu",
                              "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid", "silu",
                              "swish", "mish", "hard_shrink"), name="act_name"),
        StringVar(valid_sets=("orthogonal", "he_uniform", "he_normal", "glorot_uniform", "glorot_normal",
                              "lecun_uniform", "lecun_normal", "random_uniform", "random_normal"),
                  name="weight_initializer"),
        StringVar(valid_sets=("L2",), name="trainer"),
        FloatVar(lb=0.01, ub=100., name="alpha")
    ]
    # Initialize model
    tuner = GfoRvflTuner(problem_type="classification", bounds=PARAM_BOUNDS, cv=5, scoring="AS",
                         optimizer=model["class"], optimizer_paras=model["paras"], verbose=False, seed=seed)
    # Train the model
    tuner.fit(X=X_train, y=y_train)

    # Predict and evaluate
    y_pred = tuner.predict(X_test)
    res = tuner.best_estimator.evaluate(y_test, y_pred, list_metrics=LIST_METRICS)
    print(tuner.best_estimator)
    print(tuner.best_params)

    time_end = time.perf_counter() - time_start
    res_predict = {"model_name": model["name"], "time_taken": time_end, **res}

    # Best set of parameter
    res_params = {"model_name": model['name'], **tuner.best_params}
    return res_predict, res_params


def gridsearch_exp():
    time_start = time.perf_counter()
    seed = 42
    # Define RVFL Hyperparameter Space
    param_grid = {
        'size_hidden': list(range(5, 100)),
        'act_name': ["none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu",
                              "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid", "silu",
                              "swish", "mish", "hard_shrink"],
        'weight_initializer': ["orthogonal", "he_uniform", "he_normal", "glorot_uniform", "glorot_normal",
                              "lecun_uniform", "lecun_normal", "random_uniform", "random_normal"],
        'trainer': ["L2",],
        "alpha": list(range(1, 50)),
    }
    # Train the model
    searcher = GridSearchCV(estimator=RvflClassifier(seed=seed), param_grid=param_grid, cv=5)
    searcher.fit(X_train, y_train)
    # Predict and evaluate
    y_pred = searcher.best_estimator_.predict(X_test)
    res = searcher.best_estimator_.evaluate(y_test, y_pred, list_metrics=LIST_METRICS)

    time_end = time.perf_counter() - time_start
    res_predict = {"model_name": "GridSearchCV", "time_taken": time_end, **res}

    # Best set of parameter
    res_params = {"model_name": "GridSearchCV", **searcher.best_params_}
    return res_predict, res_params


def randomsearch_exp():
    from scipy.stats import randint, uniform

    time_start = time.perf_counter()
    seed = 42
    # Define RVFL Hyperparameter Space
    param_grid = {
        'size_hidden': randint(5, 151),
        'act_name': ["none", "relu", "leaky_relu", "celu", "prelu", "gelu", "elu", "selu",
                              "rrelu", "tanh", "hard_tanh", "sigmoid", "hard_sigmoid", "silu",
                              "swish", "mish", "hard_shrink"],
        'weight_initializer': ["orthogonal", "he_uniform", "he_normal", "glorot_uniform", "glorot_normal",
                              "lecun_uniform", "lecun_normal", "random_uniform", "random_normal"],
        'trainer': ["L2",],
        "alpha": uniform(0.01, 100),
    }
    # Train the model
    searcher = RandomizedSearchCV(estimator=RvflClassifier(seed=seed),
                                  param_distributions=param_grid, n_iter=EPOCH*POP_SIZE, cv=5, random_state=seed)
    searcher.fit(X_train, y_train)
    # Predict and evaluate
    y_pred = searcher.best_estimator_.predict(X_test)
    res = searcher.best_estimator_.evaluate(y_test, y_pred, list_metrics=LIST_METRICS)

    time_end = time.perf_counter() - time_start
    res_predict = {"model_name": "RandomizedSearchCV", "time_taken": time_end, **res}

    # Best set of parameter
    res_params = {"model_name": "RandomizedSearchCV", **searcher.best_params_}
    return res_predict, res_params


if __name__ == "__main__":
    Path(f"{PATH_SAVE}/compare").mkdir(parents=True, exist_ok=True)

    # Run trials in parallel for all models and seeds
    all_results = []
    all_best_params = []  # To store the best parameters for each trial

    res01, res02 = graforvfl_exp({"name": "RIME-RVFL", "class": "OriginalRIME", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}})
    res11, res12 = graforvfl_exp({"name": "SHADE-RVFL", "class": "OriginalSHADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}})
    res21, res22 = graforvfl_exp({"name": "INFO-RVFL", "class": "OriginalINFO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}})
    res31, res32 = gridsearch_exp()
    res41, res42 = randomsearch_exp()

    # Create DataFrames with headers
    df_result = pd.DataFrame([res01, res11, res21, res31, res41])  # Each row is a summary of metrics for a model/seed
    df_best_params = pd.DataFrame([res02, res12, res22, res32, res42])  # Each row is the best hyperparameters for a trial

    # Save DataFrames to CSV with headers
    df_result.to_csv(f"{PATH_SAVE}/compare/df_result.csv", index=False, header=True)
    df_best_params.to_csv(f"{PATH_SAVE}/compare/df_best_params.csv", index=False, header=True)
