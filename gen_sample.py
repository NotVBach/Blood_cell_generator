import torch
import numpy as np
import os
import collections
from models import Encoder, Decoder
from utils import biased_get_class, G_SM
from data_loader import load_mnist_data, preprocess_data
from config import args
import time

def generate_samples(img_file, lab_file, enc_file, dec_file, fold_idx, device, imbal):
    # Load and preprocess data
    dec_x, dec_y = preprocess_data(img_file, lab_file)
    
    # Initialize models
    encoder = Encoder(args).to(device)
    encoder.load_state_dict(torch.load(enc_file), strict=False)
    decoder = Decoder(args).to(device)
    decoder.load_state_dict(torch.load(dec_file), strict=False)
    
    encoder.eval()
    decoder.eval()
    
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    
    resx = []
    resy = []
    
    # Generate synthetic samples for classes 1 to 9
    for i in range(1, 10):
        xclass, yclass = biased_get_class(dec_x, dec_y, i)
        print(f'Class {i}: xclass shape: {xclass.shape}, yclass[0]: {yclass[0]}')
        
        # Encode xclass to feature space
        xclass = torch.Tensor(xclass).to(device)
        xclass_enc = encoder(xclass)
        xclass_enc = xclass_enc.detach().cpu().numpy()
        
        # Generate synthetic samples using SMOTE
        n = imbal[0] - imbal[i]
        xsamp, ysamp = G_SM(xclass_enc, yclass, n, i)
        print(f'Synthetic samples shape: {xsamp.shape}, labels length: {len(ysamp)}')
        ysamp = np.array(ysamp)
        print(f'ysamp shape: {ysamp.shape}')
        
        # Decode synthetic samples back to image space
        xsamp = torch.Tensor(xsamp).to(device)
        ximg = decoder(xsamp)
        ximn = ximg.detach().cpu().numpy()
        print(f'Decoded images shape: {ximn.shape}')
        
        resx.append(ximn)
        resy.append(ysamp)
    
    # Combine synthetic and real data
    resx1 = np.vstack(resx)
    resy1 = np.hstack(resy)
    print(f'Combined synthetic images shape: {resx1.shape}')
    print(f'Combined synthetic labels shape: {resy1.shape}')
    
    # Reshape for saving
    resx1 = resx1.reshape(resx1.shape[0], -1)
    dec_x1 = dec_x.reshape(dec_x.shape[0], -1)
    print(f'Reshaped synthetic images: {resx1.shape}')
    print(f'Reshaped real images: {dec_x1.shape}')
    
    combx = np.vstack((resx1, dec_x1))
    comby = np.hstack((resy1, dec_y))
    print(f'Final combined images shape: {combx.shape}')
    print(f'Final combined labels shape: {comby.shape}')
    
    # Save combined data
    ifile = f'MNIST/trn_img_f/{fold_idx}_trn_img.txt'
    lfile = f'MNIST/trn_lab_f/{fold_idx}_trn_lab.txt'
    np.savetxt(ifile, combx)
    np.savetxt(lfile, comby)
    print(f'Saved images to {ifile}')
    print(f'Saved labels to {lfile}')

def main():
    t0 = time.time()
    
    print(f'CUDA version: {torch.version.cuda}')
    
    dtrnimg = 'MNIST/trn_img/'
    dtrnlab = 'MNIST/trn_lab/'
    modpth = 'MNIST/models/crs5/'
    
    # Load file paths
    idtri_f, idtrl_f = load_mnist_data(dtrnimg, dtrnlab)
    print(idtri_f)
    print(idtrl_f)
    
    # Define model paths
    encf = [os.path.join(modpth, str(p), 'bst_enc.pth') for p in range(5)]
    decf = [os.path.join(modpth, str(p), 'bst_dec.pth') for p in range(5)]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    
    # Imbalanced sample sizes for each class
    imbal = [4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40]
    
    for m in range(5):
        print(f'\nFold {m}')
        print(f'Image file: {idtri_f[m]}')
        print(f'Label file: {idtrl_f[m]}')
        generate_samples(idtri_f[m], idtrl_f[m], encf[m], decf[m], m, device, imbal)
    
    t1 = time.time()
    print(f'Final time (min): {(t1 - t0) / 60:.2f}')

if __name__ == '__main__':
    main()