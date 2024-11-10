#!/usr/bin/env python
# Created by "Thieu" at 07:46, 05/11/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
from data_util import get_digits
from metaperceptron import MhaMlpTuner, DataTransformer


## Load data object
X_train, X_test, y_train, y_test = get_digits()

## Scaling dataset
dt = DataTransformer(scaling_methods=("minmax", ))
X_train = dt.fit_transform(X_train)
X_test = dt.transform(X_test)

# Example parameter grid of tuning hyper-parameter for Whale Optimization Algorithm-based MLP
LIST_PARAS = [
    {
        "name": "hidden_layers",
        "param": {
            'hidden_layers': [(idx, ) for idx in range(10, 100, 5)],
            'act_names': ['Tanh'],
            'dropout_rates': [None],
            'optim': ['OriginalWOA'],
            'optim_paras': [{"epoch": 750, "pop_size": 30},],
            'obj_name': ["F1S"],
            'seed': [42],
            "verbose": [False],
        }
    }, {
        "name": "activations",
        "param": {
            'hidden_layers': [25],
            'act_names': ['Tanh', "ReLU", "RReLU", "ReLU6", "Sigmoid", "CELU", "SELU", "GLU", "GELU", "LeakyReLU"],
            'dropout_rates': [None],
            'optim': ['OriginalWOA'],
            'optim_paras': [{"epoch": 750, "pop_size": 30},],
            'obj_name': ["F1S"],
            'seed': [42],
            "verbose": [False],
        }
    }, {
        "name": "dropout",
        "param": {
            'hidden_layers': [32],
            'act_names': ['Tanh'],
            'dropout_rates': [None, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
            'optim': ['OriginalWOA'],
            'optim_paras': [{"epoch": 750, "pop_size": 30},],
            'obj_name': ["F1S"],
            'seed': [42],
            "verbose": [False],
        }
    }
]

PATH_SAVE = "history"
for param_item in LIST_PARAS:
    print(f"Start hyper-parameter tuning for {param_item['name']}")

    # Initialize the tuner
    tuner = MhaMlpTuner(
        task="classification",
        param_dict=param_item["param"],
        search_method="gridsearch",  # or "randomsearch"
        scoring='F1S',
        cv=4,
        verbose=2,  # Example additional argument
        n_jobs=10  # Parallelization
    )

    # Perform tuning
    tuner.fit(X_train, y_train)
    print("Best Parameters: ", tuner.best_params_)
    print("Best Estimator: ", tuner.best_estimator_)
    y_pred = tuner.predict(X_test)
    print(tuner.best_estimator_.evaluate(y_test, y_pred, list_metrics=["AS", "PS", "RS", "F1S", "SS", "NPV"]))

    # Save GridSearchCV results to CSV
    results_df = pd.DataFrame(tuner.searcher.cv_results_)
    results_df.to_csv(f"{PATH_SAVE}/{param_item['name']}-tuning.csv", index=False)
    print(f"Finished hyper-parameter tuning for {param_item['name']}")
