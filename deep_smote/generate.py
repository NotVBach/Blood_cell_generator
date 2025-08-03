import collections
import torch
import numpy as np
import os
import time
import json
from PIL import Image
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from deep_smote.models import Encoder, Decoder
from utils.smote import G_SM, biased_get_class
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
modpth = os.path.join(base_dir, 'models', 'deep_smote')
out_img_dir = os.path.join(base_dir, 'synthetic', 'deep_smote', 'train')
out_lab_dir = os.path.join(base_dir, 'synthetic', 'deep_smote', 'annotations')

os.makedirs(out_img_dir, exist_ok=True)
os.makedirs(out_lab_dir, exist_ok=True)

# Load data
image_files, labels, bboxes, transform, data = load_data(
    base_dir, 'train', image_size=args['image_size'], n_channel=args['n_channel']
)
print(f"Loaded {len(image_files)} images, label distribution: {collections.Counter(labels)}")
print(f"Images shape: {len(image_files)}, Bboxes shape: {bboxes.shape}")

# Load images
images = np.array([transform(Image.open(p).convert('RGB')).numpy() for p in image_files])

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load pre-trained models
path_enc = os.path.join(modpth, 'bst_enc.pth')
path_dec = os.path.join(modpth, 'bst_dec.pth')
encoder = Encoder(args).to(device)
decoder = Decoder(args).to(device)
encoder.load_state_dict(torch.load(path_enc, map_location=device), strict=True)
decoder.load_state_dict(torch.load(path_dec, map_location=device), strict=True)
encoder.eval()
decoder.eval()

# Balance classes
max_samples = max(collections.Counter(labels).values())
imbal = [max_samples] * 8
resx, resy, res_bboxes = [], [], []
new_annotations = data['images'].copy()
new_anns = data['annotations'].copy()
new_image_id = max(img['id'] for img in data['images']) + 1
new_annotation_id = max(ann['id'] for ann in data['annotations']) + 1

# Generate synthetic samples
image_size_scalar = args['image_size'][0]  # Assume square images
for i in range(1, 9):
    xclass, yclass, bbox_class = biased_get_class(i, images, labels, bboxes)
    print(f"Class {i} shape: {xclass.shape}, Bboxes: {bbox_class.shape}")
    
    xclass_t = torch.tensor(xclass, dtype=torch.float32).to(device)
    bbox_class_t = torch.tensor(bbox_class, dtype=torch.float32).to(device)
    with torch.no_grad():
        z_class = encoder(xclass_t, bbox_class_t).cpu().numpy()
    
    n = imbal[i-1] - len(yclass)
    if n > 0:
        xsamp, ysamp, bbox_samp = G_SM(z_class[:, :-args['bbox_dim']], yclass, bbox_class, n, i)
        xsamp = torch.tensor(np.concatenate([xsamp, bbox_samp], axis=1), dtype=torch.float32).to(device)
        
        ximg_all, bbox_img_all = [], []
        for j in range(0, len(xsamp), args['batch_size']):
            batch = xsamp[j:j+args['batch_size']]
            with torch.no_grad():
                ximg, bbox_img = decoder(batch)
            ximg_all.append(ximg.cpu().numpy())
            bbox_img_all.append(bbox_img.cpu().numpy())
        ximn = np.concatenate(ximg_all, axis=0)
        bbox_img = np.concatenate(bbox_img_all, axis=0)
        print(f"Decoded images: {ximn.shape}, Bboxes: {bbox_img.shape}")
        
        # Save synthetic images and update annotations
        for idx, (img, bbox) in enumerate(zip(ximn, bbox_img)):
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
                'bbox': [float(x) for x in (bbox * np.array([image_size_scalar] * 4))],
                'area': float(bbox[2] * bbox[3] * image_size_scalar * image_size_scalar),
                'iscrowd': 0,
                'category_id': i,
                'segmentation': []
            })
            new_image_id += 1
            new_annotation_id += 1
        
        resx.append(ximn)
        resy.append(np.array(ysamp))
        res_bboxes.append(bbox_img)

# Combine and save
resx1 = np.vstack(resx) if resx else np.array([])
resy1 = np.hstack(resy) if resy else np.array([])
res_bboxes1 = np.vstack(res_bboxes) if res_bboxes else np.array([])
print(f"Synthetic images: {resx1.shape}, Labels: {resy1.shape}, Bboxes: {res_bboxes1.shape}")

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