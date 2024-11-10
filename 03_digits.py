#!/usr/bin/env python
# Created by "Thieu" at 21:33, 04/11/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from metaperceptron import DataTransformer, MhaMlpClassifier
from data_util import get_digits


## Load data object
X_train, X_test, y_train, y_test = get_digits()

## Scaling dataset
dt = DataTransformer(scaling_methods=("minmax", ))
X_train_scaled = dt.fit_transform(X_train)
X_test_scaled = dt.transform(X_test)

data = (X_train_scaled, X_test_scaled, y_train, y_test)
DATA_NAME = "digits"
EPOCH = 1000
POP_SIZE = 30
LIST_SEEDS = [10, 11, 12, 15, 18, 21, 24, 25, 30, 42]
LIST_METRICS = ["AS", "PS", "RS", "F1S", "SS", "NPV"]
PATH_SAVE = "history"
N_WORKERS = 10

LIST_MODELS = [
    {"name": "GA-MLP", "class": "BaseGA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "JADE-MLP", "class": "JADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "L-SHADE-MLP", "class": "L_SHADE", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "PSO-MLP", "class": "OriginalPSO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "HPSO-TVAC-MLP", "class": "HPSO_TVAC", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "P-PSO-MLP", "class": "P_PSO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "RW-GWO-MLP", "class": "RW_GWO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "AO-MLP", "class": "OriginalAO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "ARO-MLP", "class": "OriginalARO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "SMA-MLP", "class": "OriginalSMA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "AOA-MLP", "class": "OriginalAOA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "GBO-MLP", "class": "OriginalGBO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "WOA-MLP", "class": "OriginalWOA", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
    {"name": "ASO-MLP", "class": "OriginalASO", "paras": {"epoch": EPOCH, "pop_size": POP_SIZE}},
]

# Function to train, test, and evaluate a model for a single seed
def run_trial(model, seed, data):
    X_train, X_test, y_train, y_test = data

    # Initialize model
    net = MhaMlpClassifier(
        hidden_layers=(32, ), act_names="Tanh", dropout_rates=None, act_output=None,
        optim=model["class"], optim_paras=model["paras"], obj_name="F1S", seed=seed, verbose=False
    )

    # Train the model
    net.fit(X=X_train, y=y_train)

    # Collect epoch-wise training loss
    epoch_losses = [{"model_name": model["name"], "seed": seed, "epoch": epoch+1, "loss": loss}
                    for epoch, loss in enumerate(net.loss_train)]

    # Predict and evaluate
    y_pred = net.predict(X_test)
    res = net.evaluate(y_test, y_pred, list_metrics=LIST_METRICS)
    result = {"model_name": model["name"], "seed": seed, **res}
    return epoch_losses, result


if __name__ == "__main__":
    Path(f"{PATH_SAVE}/{DATA_NAME}").mkdir(parents=True, exist_ok=True)

    # Run trials in parallel for all models and seeds
    all_epoch_losses = []
    all_results = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = []
        for model in LIST_MODELS:
            for seed in LIST_SEEDS:
                futures.append(executor.submit(run_trial, model, seed, data))

        # Collect results as they complete
        for future in futures:
            epoch_losses, result = future.result()
            all_epoch_losses.extend(epoch_losses)  # Add all epoch-wise losses for this trial
            all_results.append(result)  # Add evaluation result for this trial

    # Create DataFrames with headers
    df_loss = pd.DataFrame(all_epoch_losses)  # Each row is a single epoch loss for a model/seed
    df_result = pd.DataFrame(all_results)  # Each row is a summary of metrics for a model/seed

    # Save DataFrames to CSV with headers
    df_loss.to_csv(f"{PATH_SAVE}/{DATA_NAME}/df_loss.csv", index=False, header=True)
    df_result.to_csv(f"{PATH_SAVE}/{DATA_NAME}/df_result.csv", index=False, header=True)

    print(f"Done with data: {DATA_NAME}.")
