#!/usr/bin/env python
# Created by "Thieu" at 00:11, 02/12/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import os
import pandas as pd

# Define the folder path
folder_path = 'history'

# Define the wrong and correct model names
wrong_model_name_1 = 'SHIO-SCA-RVFL'
correct_model_name_1 = 'SHIO-RVFL'
wrong_model_name_2 = 'LARO-RVFL'
correct_model_name_2 = 'IM-ARO-RVFL'

# Function to rename model names in a CSV file
def rename_models_in_csv(file_path):
    df = pd.read_csv(file_path)
    df.replace({wrong_model_name_1: correct_model_name_1, wrong_model_name_2: correct_model_name_2}, inplace=True)
    df.to_csv(file_path, index=False)

# Traverse through all files in the folder and subfolders
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)
            rename_models_in_csv(file_path)

print("Model names updated successfully!")

