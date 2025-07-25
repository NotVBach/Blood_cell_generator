import torch
import numpy as np
import os
import collections
from models import Encoder, Decoder
from utils import biased_get_class, G_SM, save_images, compute_imbal
from data_loader import preprocess_data, load_data
from config import args
import time

def generate_samples(img_dir, ann_file, enc_file, dec_file, device, imbal):
    dec_x, dec_y = preprocess_data(img_dir, ann_file)
    
    encoder = Encoder(args).to(device)
    encoder.load_state_dict(torch.load(enc_file), strict=False)
    decoder = Decoder(args).to(device)
    decoder.load_state_dict(torch.load(dec_file), strict=False)
    
    encoder.eval()
    decoder.eval()
    
    # classes = tuple(str(i) for i in range(1, args['num_classes'] + 1))
    resx = []
    resy = []
    
    for i in range(1, args['num_classes'] + 1):
        xclass, yclass = biased_get_class(dec_x, dec_y, i) # 
        print(f'Class {i}: xclass shape: {xclass.shape}, yclass[0]: {yclass[0]}')
        if len(xclass) == 0:
            print(f'Warning: No samples for class {i}, skipping...')
            continue
        
        xclass = torch.Tensor(xclass).to(device)
        print(f'xclass tensor shape: {xclass.shape}')
        xclass_enc = encoder(xclass)
        xclass_enc = xclass_enc.detach().cpu().numpy()
        
        n =  np.max(imbal) - imbal[i - 1]

        # If the class is majority
        if n == 0:
            print(f'No samples needed for class {i}, skipping...')
            continue

        xsamp, ysamp = G_SM(xclass_enc, yclass, n, i) # Generate synthetic samples
        print(f'Synthetic samples shape: {xsamp.shape}, labels length: {len(ysamp)}')
        ysamp = np.array(ysamp)
        print(f'ysamp shape: {ysamp.shape}')
        
        xsamp = torch.Tensor(xsamp).to(device)
        ximg = decoder(xsamp)
        ximn = ximg.detach().cpu().numpy()
        # print(f'Decoded images shape: {ximn.shape}')
        
        resx.append(ximn)
        resy.append(ysamp)
    
    if not resx:
        print('No synthetic samples generated')
        return
    
    resx1 = np.vstack(resx)
    resy1 = np.hstack(resy)

    output_dir = 'noaug/synthetic_images/'
    save_images(resx1, resy1, output_dir, prefix='synth')
    
    # Save as text file (Optional)

    # resx1 = resx1.reshape(resx1.shape[0], -1)
    # dec_x1 = dec_x.reshape(dec_x.shape[0], -1)
    # print(f'Reshaped synthetic images: {resx1.shape}')
    # print(f'Reshaped real images: {dec_x1.shape}')
    
    # combx = np.vstack((resx1, dec_x1))
    # comby = np.hstack((resy1, dec_y))
    # print(f'Final combined images shape: {combx.shape}')
    # print(f'Final combined labels shape: {comby.shape}')
    
    # ifile = f'noaug/trn_img_f/0_trn_img.txt'
    # lfile = f'noaug/trn_lab_f/0_trn_lab.txt'
    # np.savetxt(ifile, combx)
    # np.savetxt(lfile, comby)
    # print(f'Saved images to {ifile}')
    # print(f'Saved labels to {lfile}')

def main():
    t0 = time.time()
    
    print(f'CUDA version: {torch.version.cuda}')
    
    data_dir = 'noaug/'
    img_dir, ann_file = load_data(data_dir, split='train')
    
    print(f'Image dir: {img_dir}')
    print(f'Annotation file: {ann_file}')
    
    modpth = 'noaug/models'
    enc_file = os.path.join(modpth, 'bst_enc.pth')
    dec_file = os.path.join(modpth, 'bst_dec.pth')
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    imbal = compute_imbal(ann_file, num_classes=args['num_classes'])
    
    
    generate_samples(img_dir, ann_file, enc_file, dec_file, device, imbal)
    
    t1 = time.time()
    print(f'Final time (min): {(t1 - t0) / 60:.2f}')

if __name__ == '__main__':
    main()