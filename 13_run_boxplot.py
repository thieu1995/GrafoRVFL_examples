#!/usr/bin/env python
# Created by "Thieu" at 11:33, 26/11/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def get_figure(data_name, metrics, path_read, path_save, exts=(".png", ), verbose=False):
    Path(f"{path_save}").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(f"{path_read}/df_result.csv")

    # # Define a list of color palettes for each metric
    # color_palettes = ["Set2", "coolwarm", "Spectral", "cubehelix", "viridis", "Accent"]

    for metric in metrics:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x="model_name", y=metric, palette="Spectral", hue="model_name")
        plt.title(f"Boxplot of {metric} metric across models on {data_name} dataset", fontsize=17)
        plt.xlabel("Models", fontsize=16)
        plt.ylabel(metric, fontsize=16)
        plt.xticks(rotation=45, fontsize=14, ha="right")
        plt.yticks(fontsize=14)

        for ext in exts:
            plt.savefig(f"{path_save}/{data_name}-{metric}{ext}", bbox_inches="tight")
        if verbose:
            plt.show()


path = "history"
# Iterate through each metric and create a box plot
metric_reg = ["MAE", "RMSE", "NNSE", "WI", "R", "KGE"]
metric_cls = ["AS", "PS", "RS", "F1S", "SS", "NPV"]

get_figure(data_name="iris", metrics=metric_cls, path_read=f"{path}/iris", path_save=f"{path}/visual/boxplot")
get_figure(data_name="breast-cancer", metrics=metric_cls, path_read=f"{path}/breast_cancer", path_save=f"{path}/visual/boxplot")
get_figure(data_name="digits", metrics=metric_cls, path_read=f"{path}/digits", path_save=f"{path}/visual/boxplot")
get_figure(data_name="wine", metrics=metric_cls, path_read=f"{path}/wine", path_save=f"{path}/visual/boxplot")
get_figure(data_name="phoneme", metrics=metric_cls, path_read=f"{path}/phoneme", path_save=f"{path}/visual/boxplot")
get_figure(data_name="waveform", metrics=metric_cls, path_read=f"{path}/waveform", path_save=f"{path}/visual/boxplot")
get_figure(data_name="magic-telescope", metrics=metric_cls, path_read=f"{path}/magic_telescope", path_save=f"{path}/visual/boxplot")
get_figure(data_name="diabetes", metrics=metric_reg, path_read=f"{path}/diabetes", path_save=f"{path}/visual/boxplot")
get_figure(data_name="boston", metrics=metric_reg, path_read=f"{path}/boston", path_save=f"{path}/visual/boxplot")
get_figure(data_name="california", metrics=metric_reg, path_read=f"{path}/california", path_save=f"{path}/visual/boxplot")
