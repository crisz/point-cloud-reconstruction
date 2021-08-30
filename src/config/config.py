from pathlib import Path

# Paths
dataset_base = Path("dataset/dataset")
novel_cat_dataset_base = Path("dataset")


train_split_info = dataset_base / "train_test_split" / "shuffled_train_file_list.json"
test_split_info = dataset_base / "train_test_split" / "shuffled_test_file_list.json"
val_split_info = dataset_base / "train_test_split" / "shuffled_val_file_list.json"

label_codes = dataset_base / "synsetoffset2category.txt"

novel_categories_sim = novel_cat_dataset_base / "sncore_v1_similarCat"
novel_categories_dissim = novel_cat_dataset_base / "sncore_v1_dissimCat"
# Prediction
y_pred_path = Path(".") / "prediction.npy"

# Parameters
code_size = 1024

allowed_categories = ["02691156", "03001627", "04379243", "03636649", "02958343", "03790512", "03797390"]
# allowed_categories = ["03001627", "04379243"]
# allowed_categories = ["03797390", "03790512"]

