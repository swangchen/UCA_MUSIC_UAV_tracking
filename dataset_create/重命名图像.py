import os

# Define the path to your image folder
image_folder_path = "database\黑1\\result"

# Get a list of image files in the folder
image_files = [f for f in os.listdir(image_folder_path) if f.endswith(".png")]

# Sort the image files (optional)
image_files.sort()

# Starting index for renaming
start_index = 257


# Rename the image files starting from 91
for idx, old_name in enumerate(image_files):
    new_name = f"image_{start_index + idx}.jpg"
    old_path = os.path.join(image_folder_path, old_name)
    new_path = os.path.join(image_folder_path, new_name)

    # Check if the new file already exists, and if it does, add a suffix
    suffix = 1
    while os.path.exists(new_path):
        new_name = f"image_{start_index + idx}_{suffix}.jpg"
        new_path = os.path.join(image_folder_path, new_name)
        suffix += 1

    os.rename(old_path, new_path)

print("Image files renamed starting from 91!")
