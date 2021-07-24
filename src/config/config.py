from pathlib import Path

# Paths
dataset_base = Path("dataset/dataset")

train_split_info = dataset_base / "train_test_split" / "shuffled_train_file_list.json"
test_split_info = dataset_base / "train_test_split" / "shuffled_test_file_list.json"
val_split_info = dataset_base / "train_test_split" / "shuffled_val_file_list.json"

label_codes = dataset_base / "synsetoffset2category.txt"

# Prediction
y_pred_path = Path(".") / "prediction.npy"

# Parameters
code_size = 512
