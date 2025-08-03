import json
import os

'''
Simplify the problem by delete every images with 2 or more annotations.
'''

# Paths
json_path = '../noaug/annotations/train.json'
train_dir = '../noaug/train'

with open(json_path, 'r') as f:
    data = json.load(f)

image_annotation_count = {}
for ann in data['annotations']:
    image_id = ann['image_id']
    image_annotation_count[image_id] = image_annotation_count.get(image_id, 0) + 1

multi_annotation_image_ids = [image_id for image_id, count in image_annotation_count.items() if count > 1]

images_to_keep = [img for img in data['images'] if img['id'] not in multi_annotation_image_ids]
annotations_to_keep = [ann for ann in data['annotations'] if ann['image_id'] not in multi_annotation_image_ids]

for img in data['images']:
    if img['id'] in multi_annotation_image_ids:
        image_path = os.path.join(train_dir, img['file_name'])
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted image: {image_path}")
        else:
            print(f"Image not found: {image_path}")

updated_data = {
    'images': images_to_keep,
    'annotations': annotations_to_keep,
    'categories': data['categories'] 
}

with open(json_path, 'w') as f:
    json.dump(updated_data, f, indent=4)
    print(f"Updated JSON file: {json_path}")

print(f"Removed {len(data['images']) - len(images_to_keep)} images with multiple annotations.")
print(f"Kept {len(images_to_keep)} images and {len(annotations_to_keep)} annotations.")