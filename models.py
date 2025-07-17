import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        self.image_size = args['image_size']
        
        self.conv = nn.Sequential(
            nn.Conv2d(self.n_channel, self.dim_h, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        conv_out_size = self.image_size // (2 ** 4)
        self.fc = nn.Linear(self.dim_h * 8 * conv_out_size * conv_out_size, self.n_z)

    def forward(self, x):
        # print(f'Encoder input shape: {x.shape}')
        x = self.conv(x)
        # print(f'After conv shape: {x.shape}')
        x = x.view(x.size(0), -1)  # Explicitly flatten: [batch_size, 512 * 2 * 2]
        # print(f'After flatten shape: {x.shape}')
        x = self.fc(x)
        # print(f'After fc shape: {x.shape}')
        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        self.image_size = args['image_size']

        deconv_in_size = self.image_size // (2 ** 3)
        self.fc = nn.Sequential(
            nn.Linear(self.n_z, self.dim_h * 8 * deconv_in_size * deconv_in_size),
            nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, self.n_channel, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        # print(f'Decoder input shape: {x.shape}')
        # [x, 300]
        x = self.fc(x)
        # print(f'After fc shape: {x.shape}')
        # [x, 8192]
        x = x.view(-1, self.dim_h * 8, self.image_size // 8, self.image_size // 8)
        # print(f'After reshape shape: {x.shape}')
        # [x, 512, 4, 4]
        x = self.deconv(x)
        # print(f'After deconv shape: {x.shape}')
        # [x, 3, 32, 32]
        return x