import torch
import torch.nn as nn
import time
import numpy as np
from models import Encoder, Decoder
from utils import free_params, biased_get_class
from data_loader import load_mnist_data, preprocess_data, create_dataloader
from config import args

def train_model(encoder, decoder, dataloader, device, args, img_file, lab_file, fold_idx):
    criterion = nn.MSELoss().to(device)
    enc_optim = torch.optim.Adam(encoder.parameters(), lr=args['lr'])
    dec_optim = torch.optim.Adam(decoder.parameters(), lr=args['lr'])
    
    best_loss = np.inf
    t0 = time.time()
    
    dec_x, dec_y = preprocess_data(img_file, lab_file)
    
    for epoch in range(args['epochs']):
        train_loss = 0.0
        tmse_loss = 0.0
        tdiscr_loss = 0.0
        
        encoder.train()
        decoder.train()
        
        for images, labs in dataloader:
            encoder.zero_grad()
            decoder.zero_grad()
            images, labs = images.to(device), labs.to(device)
            labsn = labs.detach().cpu().numpy()
            
            # Forward pass
            z_hat = encoder(images)
            x_hat = decoder(z_hat)
            mse = criterion(x_hat, images)
            
            # SMOTE-like augmentation
            tc = np.random.choice(10, 1)[0]
            xbeg, ybeg = biased_get_class(dec_x, dec_y, tc)
            xlen = len(xbeg)
            nsamp = min(xlen, 100)
            ind = np.random.choice(list(range(len(xbeg))), nsamp, replace=False)
            xclass = xbeg[ind]
            yclass = ybeg[ind]
            
            xclen = len(xclass)
            xcminus = np.arange(1, xclen)
            xcplus = np.append(xcminus, 0)
            xcnew = xclass[xcplus].reshape(xclen, 1, 28, 28)
            
            xcnew = torch.Tensor(xcnew).to(device)
            xclass = torch.Tensor(xclass).to(device)
            xclass_enc = encoder(xclass)
            xclass_enc = xclass_enc.detach().cpu().numpy()
            
            xc_enc = xclass_enc[xcplus]
            xc_enc = torch.Tensor(xc_enc).to(device)
            ximg = decoder(xc_enc)
            
            mse2 = criterion(ximg, xcnew)
            comb_loss = mse2 + mse
            comb_loss.backward()
            
            enc_optim.step()
            dec_optim.step()
            
            train_loss += comb_loss.item() * images.size(0)
            tmse_loss += mse.item() * images.size(0)
            tdiscr_loss += mse2.item() * images.size(0)
        
        # Print average training statistics
        train_loss = train_loss / len(dataloader)
        tmse_loss = tmse_loss / len(dataloader)
        tdiscr_loss = tdiscr_loss / len(dataloader)
        print(f'Epoch: {epoch} \tTrain Loss: {train_loss:.6f} \tmse loss: {tmse_loss:.6f} \tmse2 loss: {tdiscr_loss:.6f}')
        
        # Save best model
        if train_loss < best_loss and args['save']:
            print('Saving..')
            path_enc = f'MNIST/models/crs5/{fold_idx}/bst_enc.pth'
            path_dec = f'MNIST/models/crs5/{fold_idx}/bst_dec.pth'
            torch.save(encoder.state_dict(), path_enc)
            torch.save(decoder.state_dict(), path_dec)
            best_loss = train_loss
    
    # Save final model
    path_enc = f'MNIST/models/crs5/{fold_idx}/f_enc.pth'
    path_dec = f'MNIST/models/crs5/{fold_idx}/f_dec.pth'
    print(path_enc)
    print(path_dec)
    torch.save(encoder.state_dict(), path_enc)
    torch.save(decoder.state_dict(), path_dec)
    
    t1 = time.time()
    print(f'total time(min): {(t1 - t0) / 60:.2f}')

def main():
    t3 = time.time()
    
    print(torch.version.cuda)
    
    dtrnimg = 'MNIST/trn_img'
    dtrnlab = 'MNIST/trn_lab'
    
    idtri_f, idtrl_f = load_mnist_data(dtrnimg, dtrnlab)
    print(idtri_f)
    print(idtrl_f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    
    for i in range(len(idtri_f)):
        print(f'\nFold {i}')
        encoder = Encoder(args).to(device)
        decoder = Decoder(args).to(device)
        
        dataloader = create_dataloader(*preprocess_data(idtri_f[i], idtrl_f[i]), 
                                     batch_size=args['batch_size'])
        
        if args['train']:
            train_model(encoder, decoder, dataloader, device, args, idtri_f[i], idtrl_f[i], i)
    
    t4 = time.time()
    print(f'final time(min): {(t4 - t3) / 60:.2f}')

if __name__ == '__main__':
    main()