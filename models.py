import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        self.bbox_dim = args['bbox_dim']
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

        # Compute final feature map size
        conv_out_size = (self.image_size[0] // 16, self.image_size[1] // 16)
        self.fc = nn.Linear(self.dim_h * 8 * conv_out_size[0] * conv_out_size[1] + self.bbox_dim, 
                           self.n_z + self.bbox_dim)

    def forward(self, x, bbox):
        # torch.Size([batch_size, n_channel, image_size, image_size])
        # torch.Size([100, 3, 64, 64])
        x = self.conv(x)

        # torch.Size([100, 512, 4, 4])
        x = x.view(x.size(0), -1)

        x = torch.cat([x, bbox], dim=1)

        # torch.Size([100, 8192])
        x = self.fc(x)

        # torch.Size([100, 300])
        # n_z = 300
        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        self.bbox_dim = args['bbox_dim']
        self.image_size = args['image_size']
        
        # Compute initial feature map size for deconv
        self.deconv_in_size = (self.image_size[0] // 16, self.image_size[1] // 16)
        
        self.fc = nn.Sequential(
            nn.Linear(self.n_z + self.bbox_dim, self.dim_h * 8 * self.deconv_in_size[0] * self.deconv_in_size[1]),
            nn.ReLU()
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.dim_h * 8, self.dim_h * 4, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 4, self.dim_h * 2, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h * 2, self.dim_h, 4, 2, 1),
            nn.BatchNorm2d(self.dim_h),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.dim_h, self.n_channel, 4, 2, 1),
            nn.Tanh()
        )

        self.bbox_head = nn.Sequential(
            nn.Linear(self.n_z + self.bbox_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.bbox_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        # torch.Size([100, 300])
        img = self.fc(x)

        # torch.Size([100, 512, 8, 8])
        img = img.view(-1, self.dim_h * 8, self.deconv_in_size[0], self.deconv_in_size[1])

        # torch.Size([100, 3, 64, 64])
        img = self.deconv(img)
        bbox = self.bbox_head(x)
        return img, bbox