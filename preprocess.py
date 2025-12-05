import os
import random
import shutil
from PIL import Image

# ----------------------------
# CONFIG
# ----------------------------
original_data_dir = "PetImages"   # your original dataset folder path
# Inside it, you should have: PetImages/cat and PetImages/dog

output_train_dir = "Sample_Train"
output_test_dir = "Sample_Test"

num_train = 1000
num_test = 100


# ----------------------------
# HELPER: remove corrupted files
# ----------------------------
def remove_corrupted_images(directory):
    num_deleted = 0
    total_files = 0

    print(f"Checking for corrupted images in {directory}...")
    for category in ["cat", "dog"]:
        total_files += 1
        category_path = os.path.join(directory, category)
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # verify image integrity
            except Exception:
                print(f"Removing corrupted file: {file_path}")
                os.remove(file_path)
                num_deleted += 1
    print(f"✅ Removed {num_deleted}/{total_files} corrupted images.\n")

# ----------------------------
# HELPER: create sample dataset
# ----------------------------
def create_sample_dataset(source_dir, train_dir, test_dir, num_train, num_test):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    for category in ["cat", "dog"]:
        print(f"Processing category: {category}")
        src_path = os.path.join(source_dir, category)
        images = os.listdir(src_path)
        random.shuffle(images)

        # select random subsets
        train_images = images[:num_train]
        test_images = images[num_train:num_train + num_test]

        # make category folders
        train_cat_dir = os.path.join(train_dir, category)
        test_cat_dir = os.path.join(test_dir, category)
        os.makedirs(train_cat_dir, exist_ok=True)
        os.makedirs(test_cat_dir, exist_ok=True)

        # copy files
        for img in train_images:
            shutil.copy(os.path.join(src_path, img), os.path.join(train_cat_dir, img))
        for img in test_images:
            shutil.copy(os.path.join(src_path, img), os.path.join(test_cat_dir, img))

        print(f"✅ {len(train_images)} train and {len(test_images)} test images copied for {category}\n")

# ----------------------------
# MAIN EXECUTION
# ----------------------------
remove_corrupted_images(original_data_dir)
create_sample_dataset(original_data_dir, output_train_dir, output_test_dir, num_train, num_test)
print("🎉 Sample dataset successfully created!")