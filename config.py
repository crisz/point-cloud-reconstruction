from pathlib import Path


dataset_base = Path("dataset")

train_split_info = dataset_base / "train_test_split" / "shuffled_train_file_list.json"
test_split_info = dataset_base / "train_test_split" / "shuffled_test_file_list.json"
val_split_info = dataset_base / "train_test_split" / "shuffled_val_file_list.json"
