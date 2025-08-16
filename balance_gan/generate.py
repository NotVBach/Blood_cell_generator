import collections
import torch
import numpy as np
import os
import time
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
from balance_gan.models import Generator
from utils.bagan import generate_balanced_inputs
from utils.smote import biased_get_class
from utils.json_utils import NumpyEncoder
from utils.dataset import load_data
from config import args

# Override args for generation
args['train'] = False
args['epochs'] = 1

# Print CUDA version
print(torch.version.cuda)

t0 = time.time()

# Paths
base_dir = args['base_dir']
dtrnimg = os.path.join(base_dir, 'train')
modpth = os.path.join(args['output_dir'], 'models')
out_img_dir = os.path.join(args['output_dir'], 'train')
out_lab_dir = os.path.join(args['output_dir'], 'annotations')

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lab_dir, exist_ok=True)

# Load data
image_files, labels, transform, data = load_data(
    base_dir, 'train', image_size=args['image_size'], n_channel=args['n_channel']
)
print(f"Loaded {len(image_files)} images, label distribution: {collections.Counter(labels)}")
print(f"Images shape: {len(image_files)}, Labels shape: {len(labels)}")

# Load images
images = np.array([transform(Image.open(p).convert('RGB')).numpy() for p in image_files])

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load pre-trained model
path_gen = os.path.join(modpth, 'bst_gen.pth')
generator = Generator(args).to(device)
generator.load_state_dict(torch.load(path_gen, map_location=device), strict=True)
generator.eval()

# Balance classes
max_samples = max(collections.Counter(labels).values())
imbal = [max_samples] * 8
resx, resy = [], []
new_annotations = data['images'].copy()
new_anns = data['annotations'].copy()
new_image_id = max(img['id'] for img in data['images']) + 1
new_annotation_id = max(ann['id'] for ann in data['annotations']) + 1

# Generate synthetic samples
for i in range(1, 9):
    xclass, yclass = biased_get_class(i, images, labels)
    print(f"Class {i} shape: {xclass.shape}, Labels: {yclass.shape}")
    
    n = imbal[i-1] - len(yclass)
    if n > 0:
        try:
            z_samp, class_labels = generate_balanced_inputs(yclass, n, i, args['n_z'])
        except ValueError as e:
            print(f"Error generating inputs for class {i}: {e}")
            continue
        
        z_samp_t = torch.tensor(z_samp, dtype=torch.float32).to(device)
        class_labels_t = torch.tensor(np.argmax(class_labels, axis=1), dtype=torch.long).to(device)
        
        ximg_all = []
        for j in range(0, len(z_samp), args['batch_size']):
            batch_z = z_samp_t[j:j+args['batch_size']]
            batch_labels = class_labels_t[j:j+args['batch_size']]
            with torch.no_grad():
                ximg = generator(batch_z, batch_labels)
            ximg_all.append(ximg.cpu().numpy())
        ximn = np.concatenate(ximg_all, axis=0)
        print(f"Generated images: {ximn.shape}")
        
        # Save synthetic images and update annotations
        for idx, img in enumerate(ximn):
            img = (img * 0.5 + 0.5).clip(0, 1)
            img = (img * 255).astype(np.uint8).transpose(1, 2, 0)
            img_pil = Image.fromarray(img)
            img_name = f"synthetic_{i}_{idx}.png"
            img_pil.save(os.path.join(out_img_dir, img_name))
            
            new_annotations.append({
                'file_name': img_name,
                'height': args['image_size'][0],
                'width': args['image_size'][1],
                'id': new_image_id
            })
            new_anns.append({
                'id': new_annotation_id,
                'image_id': new_image_id,
                'category_id': i,
                'area': 0,
                'bbox': [],
                'iscrowd': 0,
                'segmentation': []
            })
            new_image_id += 1
            new_annotation_id += 1
        
        resx.append(ximn)
        resy.append(np.array([i] * n))

# Combine and save
resx1 = np.vstack(resx) if resx else np.array([])
resy1 = np.hstack(resy) if resy else np.array([])
print(f"Synthetic images: {resx1.shape}, Labels: {resy1.shape}")

data['images'] = new_annotations
data['annotations'] = new_anns
with open(os.path.join(out_lab_dir, 'train_augmented.json'), 'w') as f:
    json.dump(data, f, indent=2, cls=NumpyEncoder)
print(f"Saved augmented annotations to {out_lab_dir}/train_augmented.json")

# Copy original images
for img in data['images']:
    img_path = os.path.join(dtrnimg, img['file_name'])
    img_name = os.path.basename(img_path)
    Image.open(img_path).save(os.path.join(out_img_dir, img_name))

t1 = time.time()
print(f"Total time (min): {(t1 - t0) / 60:.2f}")