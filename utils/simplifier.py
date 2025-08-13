import json
import os
import shutil
from PIL import Image
from collections import defaultdict

# Paths
input_dir = "noaug"
output_dir = "noaug1"
input_train_json = os.path.join(input_dir, "annotations", "train.json")
input_train_images = os.path.join(input_dir, "train")
output_train_json = os.path.join(output_dir, "annotations", "train.json")
output_train_images = os.path.join(output_dir, "train")
output_cropped_images = os.path.join(output_dir, "train_cropped")

# Create output directories
os.makedirs(output_train_images, exist_ok=True)
os.makedirs(os.path.dirname(output_train_json), exist_ok=True)
os.makedirs(output_cropped_images, exist_ok=True)

# Step 1: Load train.json
with open(input_train_json, 'r') as f:
    data = json.load(f)

# Step 2: Identify images with multiple annotations
image_id_to_annotations = defaultdict(list)
for ann in data['annotations']:
    image_id_to_annotations[ann['image_id']].append(ann)

# Images with exactly one annotation
single_annotation_image_ids = {image_id for image_id, anns in image_id_to_annotations.items() if len(anns) == 1}

# Step 3: Filter out annotations for 'platelet' (category_id: 8) and images with multiple annotations
filtered_annotations = []
for ann in data['annotations']:
    if ann['image_id'] in single_annotation_image_ids and ann['category_id'] != 8:
        filtered_annotations.append(ann)

# Step 4: Filter images that correspond to remaining annotations
filtered_image_ids = {ann['image_id'] for ann in filtered_annotations}
filtered_images = [img for img in data['images'] if img['id'] in filtered_image_ids]

# Step 5: Filter categories to exclude 'platelet' (id: 8)
filtered_categories = [cat for cat in data['categories'] if cat['id'] != 8]

# Step 6: Save new train.json
new_data = {
    "images": filtered_images,
    "annotations": filtered_annotations,
    "categories": filtered_categories
}
with open(output_train_json, 'w') as f:
    json.dump(new_data, f, indent=4)

# Step 7: Copy filtered images to noaug1/train/
for img in filtered_images:
    src_path = os.path.join(input_train_images, img['file_name'])
    dst_path = os.path.join(output_train_images, img['file_name'])
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"Warning: Image {src_path} not found, skipping.")

# Step 8: Crop filtered images using bounding boxes
image_id_to_annotation = {ann['image_id']: ann for ann in filtered_annotations}
for img in filtered_images:
    image_id = img['id']
    file_name = img['file_name']
    src_path = os.path.join(output_train_images, file_name)
    dst_path = os.path.join(output_cropped_images, file_name)

    # Check if image exists
    if not os.path.exists(src_path):
        print(f"Warning: Image {src_path} not found, skipping cropping.")
        continue

    # Get bounding box
    if image_id not in image_id_to_annotation:
        print(f"Warning: No annotation found for image {file_name}, skipping cropping.")
        continue

    bbox = image_id_to_annotation[image_id]['bbox']  # [x_min, y_min, width, height]
    x_min, y_min, width, height = bbox

    # Open and crop image
    try:
        with Image.open(src_path) as image:
            # Ensure bbox is within image boundaries
            img_width, img_height = image.size
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_min + width)
            y_max = min(img_height, y_min + height)

            # Crop image
            cropped_image = image.crop((x_min, y_min, x_max, y_max))

            # Save cropped image
            cropped_image.save(dst_path)
            print(f"Cropped and saved: {dst_path}")
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

print(f"Filtering and cropping complete.")
print(f"Filtered dataset saved to {output_dir}")
print(f"Cropped images saved to {output_cropped_images}")
print(f"Images: {len(filtered_images)}, Annotations: {len(filtered_annotations)}, Categories: {len(filtered_categories)}")