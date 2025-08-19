import torch
import torch.nn as nn
import math

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        self.image_size = args['image_size']
        
        self.final_dim = (self.image_size[0] // 16, self.image_size[1] // 16)
        
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
        self.pool = nn.AdaptiveAvgPool2d((4, 4))  # Ensure consistent output size
        self.fc = nn.Linear(self.dim_h * 8 * 4 * 4, self.n_z)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        self.image_size = args['image_size']
        self.num_classes = args['num_classes']
        
        self.initial_dim = (self.image_size[0] // 16, self.image_size[1] // 16)
        
        self.fc = nn.Sequential(
            nn.Linear(self.n_z + self.num_classes, self.dim_h * 8 * self.initial_dim[0] * self.initial_dim[1]),
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

    def forward(self, z, class_label):
        # if not (class_label >= 0).all() or not (class_label < self.num_classes).all():
        #     raise ValueError(f"Class labels must be in [0, {self.num_classes-1}], got {class_label}")
        # if z.size(0) != class_label.size(0):
        #     raise ValueError(f"Batch size mismatch: z ({z.size(0)}) vs class_label ({class_label.size(0)})")
        class_onehot = torch.zeros(class_label.size(0), self.num_classes, device=class_label.device)
        class_onehot.scatter_(1, class_label.unsqueeze(1), 1)
        input_vec = torch.cat([z, class_onehot], dim=1)
        img = self.fc(input_vec)
        img = img.view(-1, self.dim_h * 8, self.initial_dim[0], self.initial_dim[1])
        img = self.deconv(img)
        return img

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.n_channel = args['n_channel']
        self.ndf = args['ndf']
        self.image_size = args['image_size']
        
        self.main = nn.Sequential(
            nn.Conv2d(self.n_channel, self.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # Ensure consistent output size
            nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1)