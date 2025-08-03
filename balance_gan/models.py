import torch
import torch.nn as nn
import math

class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.n_z = args['n_z']
        self.bbox_dim = args['bbox_dim']
        self.image_size = args['image_size']
        self.num_classes = args['num_classes']
        
        self.initial_dim = (self.image_size[0] // 16, self.image_size[1] // 16)
        
        self.fc = nn.Sequential(
            nn.Linear(self.n_z + self.bbox_dim + self.num_classes, self.dim_h * 8 * self.initial_dim[0] * self.initial_dim[1]),
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
            nn.Linear(self.n_z + self.bbox_dim + self.num_classes, 128),
            nn.ReLU(),
            nn.Linear(128, self.bbox_dim),
            nn.Sigmoid()
        )

    def forward(self, z, class_label, bbox):
        # One-hot encode class label
        class_onehot = torch.zeros(class_label.size(0), self.num_classes, device=class_label.device)
        class_onehot.scatter_(1, class_label.unsqueeze(1), 1)
        # Concatenate noise, class, and bbox
        input_vec = torch.cat([z, class_onehot, bbox], dim=1)
        img = self.fc(input_vec)
        img = img.view(-1, self.dim_h * 8, self.initial_dim[0], self.initial_dim[1])
        img = self.deconv(img)
        bbox_pred = self.bbox_head(input_vec)
        return img, bbox_pred

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.n_channel = args['n_channel']
        self.dim_h = args['dim_h']
        self.image_size = args['image_size']
        self.num_classes = args['num_classes']
        
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
        self.fc_real = nn.Linear(self.dim_h * 8 * self.final_dim[0] * self.final_dim[1], 1)
        self.fc_class = nn.Linear(self.dim_h * 8 * self.final_dim[0] * self.final_dim[1], self.num_classes)
        self.fc_bbox = nn.Sequential(
            nn.Linear(self.dim_h * 8 * self.final_dim[0] * self.final_dim[1], 128),
            nn.ReLU(),
            nn.Linear(128, args['bbox_dim']),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        real_score = self.fc_real(x)
        class_score = self.fc_class(x)
        bbox_pred = self.fc_bbox(x)
        return real_score, class_score, bbox_pred