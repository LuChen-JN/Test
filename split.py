import pandas as pd
import os
from sklearn.model_selection import train_test_split, KFold

# Define the file path
file_path = './datasets/RNA/all.csv'  # Replace with the actual path
output_dir = './datasets'

# 1. Load the data
data = pd.read_csv(file_path)

# 2. Divide the data set
train_val_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# 3. 5-fold cross-validation divides the training set into the validation set
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create an output folder
os.makedirs(output_dir, exist_ok=True)

# 4. Save the data for each fold
fold = 1
for train_idx, val_idx in kf.split(train_val_data):       # data   train_val_data
    # Create folders
    fold_dir = os.path.join(output_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    
    # Get the training set and validation set
    train_fold = train_val_data.iloc[train_idx]     # data   train_val_data
    val_fold = train_val_data.iloc[val_idx]         # data   train_val_data
    
    # Save the data to the corresponding folder
    train_fold.to_csv(os.path.join(fold_dir, "train.csv"), index=False)
    val_fold.to_csv(os.path.join(fold_dir, "val.csv"), index=False)
    test_data.to_csv(os.path.join(fold_dir, "test.csv"), index=False)
    
    fold += 1

print(f"The data has been successfully saved to {output_dir}")
