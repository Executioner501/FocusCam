import os
import shutil

# Paths — change these to your actual directories
source_folder = r"E:\user06\user06\YawDD dataset\YawDD dataset\Mirror\Female_mirror"
yawning_folder = r"E:\yawn"
others_folder = r"E:\no_yawn"

# Make sure destination folders exist
os.makedirs(yawning_folder, exist_ok=True)
os.makedirs(others_folder, exist_ok=True)

# Loop through files in source folder
for filename in os.listdir(source_folder):
    # Full path of the file
    file_path = os.path.join(source_folder, filename)

    # Skip directories, only process files
    if os.path.isfile(file_path):
        # Check if "Yawning" is in the file name (case-insensitive)
        if "yawning" in filename.lower():
            shutil.copy(file_path, yawning_folder)
            print(f"Copied: {filename} → Yawning folder")
        else:
            shutil.copy(file_path, others_folder)
            print(f"Copied: {filename} → Others folder")
